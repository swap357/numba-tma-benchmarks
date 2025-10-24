#!/usr/bin/env python3
"""
Collect TMA (Top-down Microarchitecture Analysis) metrics for Numba benchmarks
"""
import subprocess
import json
import csv
import sys
import re
from collections import defaultdict

# Benchmark configurations
BENCHMARKS = [
    ("sum1d", 1000000),
    ("sum2d", 1000),
    ("while_count", 1000000),
    ("andor", 1000),
    ("copy_arrays", 1000000),
    ("copy_arrays2d", 1000),
    ("blackscholes_cnd", 10000),
]

ENGINES = ["py", "jit"]

def run_perf_stat(function, engine, n, reps=3, cpu_pin="0"):
    """
    Run perf stat and collect TMA metrics with proper benchmarking methodology
    
    Args:
        function: Function to benchmark
        engine: 'py' or 'jit'
        n: Problem size
        reps: Number of repetitions
        cpu_pin: CPU core(s) to pin to (default: single P-core for clean metrics)
    
    Note: For hybrid architectures, pinning to a SINGLE P-core gives cleanest
    TMA metrics. Using multiple cores mixes counters and breaks the L1 decomposition.
    """
    cmd = [
        "taskset", "-c", cpu_pin,  # Pin process to specific CPU(s)
        "perf", "stat",
        "-x,",  # CSV output
        "-a",   # System-wide (but process is pinned via taskset)
        "-M", "tma_L1_group,tma_L2_group,tma_L3_group",
        "--",  # Separate perf args from command
        "python", "benchmark_usecases.py",
        "-f", function,
        "-e", engine,
        "-n", str(n),
        "--reps", str(reps)
    ]
    
    print(f"Benchmarking {function:15s} [{engine:3s}] n={n:>8d}", 
          file=sys.stderr, end=' ... ', flush=True)
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/numba-experiments")
    
    # Parse stdout for benchmark result (JSON)
    benchmark_result = None
    for line in result.stdout.strip().split('\n'):
        try:
            benchmark_result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    
    # Parse stderr for perf stats (only P-core metrics)
    metrics = parse_perf_output(result.stderr, prefer_pcore=True)
    
    return benchmark_result, metrics

def parse_perf_output(perf_stderr, prefer_pcore=True):
    """
    Parse perf stat CSV output and extract TMA metrics
    
    Args:
        perf_stderr: stderr output from perf stat
        prefer_pcore: If True, only extract cpu_core metrics (ignore cpu_atom)
    
    Returns:
        dict: TMA metrics with their values
    """
    metrics = {}
    
    # Look for lines with TMA metrics (they have % and "tma_" in them)
    for line in perf_stderr.split('\n'):
        if not line.strip() or line.startswith('#') or 'tma_' not in line:
            continue
        
        # Filter out E-core (cpu_atom) metrics if prefer_pcore is True
        if prefer_pcore and 'cpu_atom' in line:
            continue
        
        parts = line.split(',')
        if len(parts) < 6:
            continue
        
        # Format: value1,value2,counter,value4,value5,metric_value,metric_name
        # or: ,,,,,,metric_value,metric_name
        # Last column typically has the metric name
        metric_name = parts[-1].strip()
        
        if 'tma_' in metric_name:
            # Clean up metric name (remove %, whitespace)
            metric_name = metric_name.replace('%', '').strip()
            
            # Skip if we already have this metric (avoid duplicates from multiple cores)
            # Keep first occurrence (usually the most relevant one)
            if metric_name in metrics:
                continue
            
            # Value is usually in column 5 or 6 (0-indexed)
            # Try to find a numeric value with or without %
            value = None
            for col in parts[-3:-1]:  # Check last few columns before metric name
                col = col.strip().replace('%', '').strip()
                if col and col not in ['', 'umask=0x03', 'umask=0x3c', 'umask=0xc', 'umask=0x80', 'cmask=1', 'edge']:
                    try:
                        value = float(col)
                        break
                    except ValueError:
                        continue
            
            if value is not None and metric_name:
                metrics[metric_name] = value
    
    return metrics

def setup_benchmarking_environment():
    """Set up environment for reproducible benchmarking"""
    import os
    
    # Set process priority (requires root)
    try:
        os.nice(-10)
    except:
        pass  # Continue without elevated priority

def collect_all_metrics(cpu_pin="0", reps=5):
    """
    Collect TMA metrics for all benchmarks with proper methodology
    
    Args:
        cpu_pin: CPU core to pin to (default: "0", single P-core for clean TMA metrics)
        reps: Number of repetitions per benchmark (default: 5)
    
    Note: Single-core pinning is ESSENTIAL for correct TMA L1 metrics.
    Multi-core pinning causes counter aggregation issues that break the decomposition.
    """
    setup_benchmarking_environment()
    results = []
    
    print(f"CPU: {cpu_pin} | Reps: {reps} | Total: {len(BENCHMARKS) * len(ENGINES)} runs\n", 
          file=sys.stderr)
    
    for function, n in BENCHMARKS:
        for engine in ENGINES:
            try:
                bench_result, tma_metrics = run_perf_stat(function, engine, n, reps=reps, cpu_pin=cpu_pin)
                
                if bench_result and tma_metrics:
                    # Combine benchmark info with TMA metrics
                    elapsed_times = bench_result.get('elapsed_ns_each', [])
                    row = {
                        'function': function,
                        'engine': engine,
                        'n': n,
                        'reps': bench_result.get('reps', 0),
                        'elapsed_ns_mean': sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0,
                        'elapsed_ns_min': min(elapsed_times) if elapsed_times else 0,
                        'elapsed_ns_max': max(elapsed_times) if elapsed_times else 0,
                        'elapsed_ns_std': (sum((x - sum(elapsed_times)/len(elapsed_times))**2 for x in elapsed_times) / len(elapsed_times))**0.5 if len(elapsed_times) > 1 else 0,
                    }
                    row.update(tma_metrics)
                    results.append(row)
                    print(f"✓", file=sys.stderr)
                else:
                    print(f"✗ Failed", file=sys.stderr)
                    
            except Exception as e:
                print(f"✗ Error: {e}", file=sys.stderr)
                continue
    
    return results

def write_csv(results, output_file):
    """Write results to CSV file"""
    if not results:
        print("No results to write!", file=sys.stderr)
        return
    
    # Get all unique metric names
    all_fields = set()
    for row in results:
        all_fields.update(row.keys())
    
    # Sort fields: put basic info first, then TMA metrics
    basic_fields = ['function', 'engine', 'n', 'reps', 'elapsed_ns_mean']
    tma_fields = sorted([f for f in all_fields if f.startswith('tma_')])
    other_fields = sorted([f for f in all_fields if f not in basic_fields and not f.startswith('tma_')])
    
    fieldnames = basic_fields + tma_fields + other_fields
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved {len(results)} results to {output_file}", file=sys.stderr)
    print(f"  {len(tma_fields)} TMA metrics collected per benchmark", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect TMA metrics for Numba benchmarks with proper methodology"
    )
    parser.add_argument(
        "output", 
        nargs="?", 
        default="tma_results.csv",
        help="Output CSV file (default: tma_results.csv)"
    )
    parser.add_argument(
        "--cpu-pin", 
        default="0",
        help="CPU core to pin to (default: 0, must be a single P-core for accurate TMA)"
    )
    parser.add_argument(
        "--reps", 
        type=int, 
        default=5,
        help="Number of repetitions per benchmark (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("TMA Metrics Collection", file=sys.stderr)
    print("="*50, file=sys.stderr)
    
    results = collect_all_metrics(cpu_pin=args.cpu_pin, reps=args.reps)
    write_csv(results, args.output)

