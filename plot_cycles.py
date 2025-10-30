#!/usr/bin/env python3
"""TMA Metrics Visualization - Absolute CPU Cycles"""
import sys
import subprocess
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_cpu_frequency():
    """Get CPU max frequency from system in GHz"""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'CPU max MHz' in line:
                return float(line.split(':')[1].strip()) / 1000.0
    except:
        pass
    print("Warning: Could not detect CPU frequency, assuming 4.0 GHz", file=sys.stderr)
    return 4.0

CPU_FREQ_GHZ = get_cpu_frequency()

TMA_HIERARCHY = {
    'l1': ['tma_retiring', 'tma_bad_speculation', 'tma_frontend_bound', 'tma_backend_bound'],
    'l2': {
        'retiring': ['tma_light_operations', 'tma_heavy_operations'],
        'bad_speculation': ['tma_branch_mispredicts', 'tma_machine_clears'],
        'frontend': ['tma_fetch_latency', 'tma_fetch_bandwidth'],
        'backend': ['tma_memory_bound', 'tma_core_bound'],
    },
    'l3': {
        'light_operations': ['tma_fp_arith', 'tma_int_operations', 'tma_fused_instructions', 
                             'tma_memory_operations', 'tma_non_fused_branches', 'tma_nop_instructions'],
        'heavy_operations': ['tma_few_uops_instructions', 'tma_microcode_sequencer'],
        'memory_bound': ['tma_l1_bound', 'tma_l2_bound', 'tma_l3_bound', 'tma_dram_bound', 'tma_store_bound'],
        'core_bound': ['tma_divider', 'tma_ports_utilization'],
    }
}

L1_COLORS = {
    'tma_retiring': '#2ecc71',
    'tma_bad_speculation': '#e74c3c', 
    'tma_frontend_bound': '#3498db',
    'tma_backend_bound': '#f39c12',
}

L2_COLORS = {
    'retiring': '#2ecc71', 'bad_speculation': '#e74c3c', 
    'frontend': '#3498db', 'backend': '#f39c12'
}

L3_COLORS = {
    'light_operations': '#27ae60', 'heavy_operations': '#229954',
    'memory_bound': '#95a5a6', 'core_bound': '#7f8c8d',
}

def ns_to_cycles(ns):
    """Convert nanoseconds to CPU cycles"""
    return ns * CPU_FREQ_GHZ

def format_cycles(cycles):
    """Format cycle count for display"""
    if cycles >= 1e6:
        return f"{cycles/1e6:.1f}M"
    elif cycles >= 1e3:
        return f"{cycles/1e3:.1f}K"
    else:
        return f"{cycles:.0f}"

def format_time(ns):
    """Format time in appropriate unit (ns, µs, ms, s)"""
    if ns >= 1e9:
        return f"{ns/1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f}µs"
    else:
        return f"{ns:.0f}ns"

def format_label(metric_name):
    """Convert tma_metric_name to 'Metric Name'"""
    return metric_name.replace('tma_', '').replace('_', ' ').title()

def add_bar_trace(fig, engine, metric, pct, total_cycles, row, color, opacity=1.0, min_cycles=500, fontsize=10, show_in_legend=False):
    """Helper to add a bar trace with cycle counts"""
    y_label = "Python" if engine == 'py' else "JIT"
    label = format_label(metric)
    cycles = (pct / 100.0) * total_cycles

    text = f"{format_cycles(cycles)} ({pct:.1f}%)" if cycles > min_cycles else ""

    fig.add_trace(go.Bar(
        name=label,
        x=[cycles],
        y=[y_label],
        orientation='h',
        marker=dict(color=color, line=dict(color='white', width=0.5), opacity=opacity),
        text=text,
        textposition='inside',
        textfont=dict(size=fontsize, color='white'),
        hovertemplate=f"<b>{label}</b><br>{y_label}<br>Cycles: {cycles:,.0f}<br>{pct:.1f}%<extra></extra>",
        showlegend=show_in_legend,
        legendgroup=label,
    ), row=row, col=1)

def plot_full_hierarchy(df, function):
    """Create comprehensive view with L1, L2, and L3 metrics in absolute cycles"""
    func_data = df[df['function'] == function]

    if len(func_data) == 0:
        print(f"Error: Function '{function}' not found")
        print(f"Available: {', '.join(df['function'].unique())}")
        sys.exit(1)

    # Calculate absolute cycles
    data = {}
    for engine in ['py', 'jit']:
        row = func_data[func_data['engine'] == engine].iloc[0]
        time_ns = row['elapsed_ns_mean']
        data[engine] = {
            'time_ns': time_ns,
            'time_str': format_time(time_ns),
            'total_cycles': ns_to_cycles(time_ns),
            'row': row
        }

    speedup = data['py']['time_ns'] / data['jit']['time_ns']
    use_log = speedup > 100

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.25, 0.4, 0.35],
        subplot_titles=(
            '<b>L1: Top-level Categories (Absolute Cycles)</b>',
            '<b>L2: Breakdown by Category (Absolute Cycles)</b>',
            '<b>L3: Detailed Breakdown (Absolute Cycles)</b>'
        ),
        vertical_spacing=0.12,
        specs=[[{'type': 'bar'}], [{'type': 'bar'}], [{'type': 'bar'}]]
    )

    engines = ['py', 'jit']

    # L1 METRICS (show in legend only for first engine)
    for i, engine in enumerate(engines):
        row = data[engine]['row']
        total = data[engine]['total_cycles']
        for metric in TMA_HIERARCHY['l1']:
            if metric in row:
                add_bar_trace(fig, engine, metric, row[metric], total, row=1,
                             color=L1_COLORS.get(metric, '#95a5a6'), min_cycles=1000, 
                             show_in_legend=(i == 0))

    # L2 METRICS
    for engine in engines:
        row = data[engine]['row']
        total = data[engine]['total_cycles']
        for cat, metrics in TMA_HIERARCHY['l2'].items():
            for metric in metrics:
                if metric in row:
                    add_bar_trace(fig, engine, metric, row[metric], total, row=2,
                                 color=L2_COLORS.get(cat, '#95a5a6'), opacity=0.8, min_cycles=500, fontsize=9)

    # L3 METRICS (only significant: >1% OR >100 cycles)
    for engine in engines:
        row = data[engine]['row']
        total = data[engine]['total_cycles']
        for cat, metrics in TMA_HIERARCHY['l3'].items():
            for metric in metrics:
                if metric in row:
                    cycles = (row[metric] / 100.0) * total
                    if row[metric] > 1.0 or cycles > 100:
                        add_bar_trace(fig, engine, metric, row[metric], total, row=3,
                                     color=L3_COLORS.get(cat, '#95a5a6'), opacity=0.7, min_cycles=300, fontsize=8)

    x_axis_title = "CPU Cycles (log scale)" if use_log else "CPU Cycles"

    fig.update_layout(
        title=f"<b>{function}</b> - Complete TMA Hierarchy (Absolute Cycles @ {CPU_FREQ_GHZ:.1f} GHz)<br>" +
              f"<sub>Python: {data['py']['time_str']} ({format_cycles(data['py']['total_cycles'])} cycles) | " +
              f"JIT: {data['jit']['time_str']} ({format_cycles(data['jit']['total_cycles'])} cycles) | " +
              f"Speedup: {speedup:.1f}x</sub>",
        barmode='stack',
        height=900,
        width=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="L1 Categories",
        ),
    )

    if use_log:
        fig.update_xaxes(type='log', title_text=x_axis_title)
    else:
        fig.update_xaxes(title_text=x_axis_title)

    fig.update_yaxes(categoryorder='array', categoryarray=['JIT', 'Python'])

    return fig

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_cycles.py <csv_file> <function> [output_file]")
        print("Example: python plot_cycles.py results_tma.csv sum1d")
        print("Example: python plot_cycles.py results_tma.csv sum1d plots/sum1d.svg")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    fig = plot_full_hierarchy(df, sys.argv[2])
    
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]
        fig.write_image(output_file, width=1400, height=900, scale=2)
        print(f"Saved plot to {output_file}")
    else:
        fig.show()

if __name__ == "__main__":
    main()

