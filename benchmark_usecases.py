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

def andor(x_arr, y_arr):
    """Boolean logic test on arrays"""
    result = 0
    for i in range(x_arr.shape[0]):
        if (x_arr[i] > 0 and x_arr[i] < 10) or (y_arr[i] > 0 and y_arr[i] < 10):
            result += 1
    return result

def copy_arrays(a, b):
    """Copy 1D array elements"""
    for i in range(a.shape[0]):
        b[i] = a[i]

def copy_arrays2d(a, b):
    """Copy 2D array elements"""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]

def blackscholes_cnd(d_arr):
    """Black-Scholes cumulative normal distribution on array"""
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    result = 0.0
    for i in range(d_arr.shape[0]):
        d = d_arr[i]
        K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
        ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
                   (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
        if d > 0:
            ret_val = 1.0 - ret_val
        result += ret_val
    return result

def matmul_naive(A, B, C):
    """Naive matrix multiplication: C = A @ B"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    for i in range(n):
        for j in range(m):
            c = 0.0
            for l in range(k):
                c += A[i, l] * B[l, j]
            C[i, j] = c

def matmul_blocked(A, B, C, block_size=32):
    """Blocked/tiled matrix multiplication: C = A @ B"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    for ii in range(0, n, block_size):
        for jj in range(0, m, block_size):
            for kk in range(0, k, block_size):
                # Process block
                i_end = min(ii + block_size, n)
                j_end = min(jj + block_size, m)
                k_end = min(kk + block_size, k)
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        c = C[i, j]  # Start with existing value (accumulates across kk blocks)
                        for l in range(kk, k_end):
                            c += A[i, l] * B[l, j]
                        C[i, j] = c

def matmul_transpose(A, B_T, C):
    """Matrix multiplication with pre-transposed B: C = A @ B_T.T"""
    # B_T is already transposed, so we access it as B_T[j, l] instead of B[l, j]
    n = A.shape[0]
    m = B_T.shape[0]  # B_T.shape[0] == original B.shape[1]
    k = A.shape[1]
    for i in range(n):
        for j in range(m):
            c = 0.0
            for l in range(k):
                c += A[i, l] * B_T[j, l]  # B_T[j, l] == original B[l, j]
            C[i, j] = c

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
matmul_naive_jit = jit(nopython=True, cache=True)(matmul_naive)
matmul_blocked_jit = jit(nopython=True, cache=True)(matmul_blocked)
matmul_transpose_jit = jit(nopython=True, cache=True)(matmul_transpose)

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
        'args_fn': lambda n: (
            ((np.arange(n, dtype=np.int32) % 20) - 10),
            (((np.arange(n, dtype=np.int32) * 7) % 20) - 10)
        ),
        'warmup_args_fn': lambda n: (
            ((np.arange(min(1000, n), dtype=np.int32) % 20) - 10),
            (((np.arange(min(1000, n), dtype=np.int32) * 7) % 20) - 10)
        ),
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
        'args_fn': lambda n: (
            ((np.arange(n, dtype=np.int32) % 200) - 100).astype(np.float64) / 10.0,
        ),
        'warmup_args_fn': lambda n: (
            ((np.arange(min(1000, n), dtype=np.int32) % 200) - 100).astype(np.float64) / 10.0,
        ),
    },
    'matmul_naive': {
        'py': matmul_naive,
        'jit': matmul_naive_jit,
        'timer': timethis_array,
        'args_fn': lambda n: (
            np.random.rand(n, n).astype(np.float64),
            np.random.rand(n, n).astype(np.float64),
            np.zeros((n, n), dtype=np.float64)
        ),
        'warmup_args_fn': lambda n: (
            np.random.rand(50, 50).astype(np.float64),
            np.random.rand(50, 50).astype(np.float64),
            np.zeros((50, 50), dtype=np.float64)
        ),
    },
    'matmul_blocked': {
        'py': matmul_blocked,
        'jit': matmul_blocked_jit,
        'timer': timethis_array,
        'args_fn': lambda n: (
            np.random.rand(n, n).astype(np.float64),
            np.random.rand(n, n).astype(np.float64),
            np.zeros((n, n), dtype=np.float64)
        ),
        'warmup_args_fn': lambda n: (
            np.random.rand(50, 50).astype(np.float64),
            np.random.rand(50, 50).astype(np.float64),
            np.zeros((50, 50), dtype=np.float64)
        ),
    },
    'matmul_transpose': {
        'py': matmul_transpose,
        'jit': matmul_transpose_jit,
        'timer': timethis_array,
        'args_fn': lambda n: (
            np.random.rand(n, n).astype(np.float64),
            np.random.rand(n, n).astype(np.float64).T,  # Pre-transpose B
            np.zeros((n, n), dtype=np.float64)
        ),
        'warmup_args_fn': lambda n: (
            np.random.rand(50, 50).astype(np.float64),
            np.random.rand(50, 50).astype(np.float64).T,  # Pre-transpose B
            np.zeros((50, 50), dtype=np.float64)
        ),
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

