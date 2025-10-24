# usecases_bench.py
import argparse, json, time, os, math
import numpy as np
from numba import jit

# ============================================================================
# Use case implementations
# ============================================================================

def sum1d(s, e):
    """Sum integers from s to e"""
    c = 0
    for i in range(s, e):
        c += i
    return c

def sum2d(s, e):
    """Nested loop sum"""
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c

def while_count(s, e):
    """While loop sum"""
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c

def andor(x, y):
    """Boolean logic test"""
    return (x > 0 and x < 10) or (y > 0 and y < 10)

def copy_arrays(a, b):
    """Copy 1D array elements"""
    for i in range(a.shape[0]):
        b[i] = a[i]

def copy_arrays2d(a, b):
    """Copy 2D array elements"""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]

def blackscholes_cnd(d):
    """Black-Scholes cumulative normal distribution"""
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val

# ============================================================================
# JIT compiled versions
# ============================================================================

sum1d_jit = jit(nopython=True, cache=True)(sum1d)
sum2d_jit = jit(nopython=True, cache=True)(sum2d)
while_count_jit = jit(nopython=True, cache=True)(while_count)
andor_jit = jit(nopython=True, cache=True)(andor)
copy_arrays_jit = jit(nopython=True, cache=True)(copy_arrays)
copy_arrays2d_jit = jit(nopython=True, cache=True)(copy_arrays2d)
blackscholes_cnd_jit = jit(nopython=True, cache=True)(blackscholes_cnd)

# ============================================================================
# Timing utilities
# ============================================================================

def timethis_scalar(f):
    """Timer for scalar functions (sum1d, sum2d, etc.)"""
    def wrapped(*args):
        t0 = time.perf_counter_ns()
        out = f(*args)
        t1 = time.perf_counter_ns()
        return out, t1 - t0
    return wrapped

def timethis_array(f):
    """Timer for array functions (copy_arrays, etc.)"""
    def wrapped(*args):
        t0 = time.perf_counter_ns()
        f(*args)
        t1 = time.perf_counter_ns()
        return None, t1 - t0
    return wrapped

# ============================================================================
# Benchmark configurations
# ============================================================================

BENCHMARKS = {
    'sum1d': {
        'py': sum1d,
        'jit': sum1d_jit,
        'timer': timethis_scalar,
        'args_fn': lambda n: (0, n),
        'warmup_args_fn': lambda n: (0, min(1000, n)),
    },
    'sum2d': {
        'py': sum2d,
        'jit': sum2d_jit,
        'timer': timethis_scalar,
        'args_fn': lambda n: (0, n),
        'warmup_args_fn': lambda n: (0, min(100, n)),  # Smaller for nested loop
    },
    'while_count': {
        'py': while_count,
        'jit': while_count_jit,
        'timer': timethis_scalar,
        'args_fn': lambda n: (0, n),
        'warmup_args_fn': lambda n: (0, min(1000, n)),
    },
    'andor': {
        'py': andor,
        'jit': andor_jit,
        'timer': timethis_scalar,
        'args_fn': lambda n: (n % 20 - 10, n % 20 - 10),
        'warmup_args_fn': lambda n: (5, 5),
    },
    'copy_arrays': {
        'py': copy_arrays,
        'jit': copy_arrays_jit,
        'timer': timethis_array,
        'args_fn': lambda n: (np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32)),
        'warmup_args_fn': lambda n: (np.arange(1000, dtype=np.int32), np.zeros(1000, dtype=np.int32)),
    },
    'copy_arrays2d': {
        'py': copy_arrays2d,
        'jit': copy_arrays2d_jit,
        'timer': timethis_array,
        'args_fn': lambda n: (
            np.arange(n*n, dtype=np.int32).reshape(n, n),
            np.zeros((n, n), dtype=np.int32)
        ),
        'warmup_args_fn': lambda n: (
            np.arange(100, dtype=np.int32).reshape(10, 10),
            np.zeros((10, 10), dtype=np.int32)
        ),
    },
    'blackscholes_cnd': {
        'py': blackscholes_cnd,
        'jit': blackscholes_cnd_jit,
        'timer': timethis_scalar,
        'args_fn': lambda n: (float(n % 100) / 100.0,),
        'warmup_args_fn': lambda n: (0.5,),
    },
}

# ============================================================================
# Main benchmark runner
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--function", "-f", choices=list(BENCHMARKS.keys()), required=True,
                   help="Function to benchmark")
    p.add_argument("--engine", "-e", choices=["jit", "py"], default="jit",
                   help="Engine to use")
    p.add_argument("-n", type=int, default=10000,
                   help="Problem size parameter")
    p.add_argument("--reps", type=int, default=5,
                   help="Number of repetitions")
    args = p.parse_args()

    # lock numpy to 1 thread if you want fairness
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Get benchmark configuration
    bench = BENCHMARKS[args.function]
    func = bench[args.engine]
    timer = bench['timer']
    
    # Prepare function with timer
    runner = timer(func)
    
    # Warmup with appropriate size
    warmup_args = bench['warmup_args_fn'](args.n)
    _ = runner(*warmup_args)
    
    # Prepare actual benchmark arguments
    benchmark_args = bench['args_fn'](args.n)
    
    # Run benchmark
    times = []
    result = None
    for _ in range(args.reps):
        val, dt = runner(*benchmark_args)
        times.append(dt)
        if result is None:
            result = val

    # Output results
    output = {
        "function": args.function,
        "engine": args.engine,
        "n": args.n,
        "reps": args.reps,
        "elapsed_ns_each": times
    }
    
    if result is not None:
        output["result"] = float(result) if isinstance(result, (int, float, np.number)) else str(result)
    
    print(json.dumps(output))

