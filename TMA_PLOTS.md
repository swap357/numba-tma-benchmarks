# TMA Metrics Visualization

Top-down Microarchitecture Analysis (TMA) hierarchy for all benchmark functions, showing absolute CPU cycles and execution times in appropriate units (ns, µs, ms, or s). Each plot includes a color legend and calculates speedup as: `Python time / JIT time`.

#### Legend

**L1 Categories (Top-level):**
- 🟢 **Retiring** (`#2ecc71`) - Useful work, instructions successfully retired
- 🔴 **Bad Speculation** (`#e74c3c`) - Wasted work from branch mispredicts and pipeline clears
- 🔵 **Frontend Bound** (`#3498db`) - CPU starved waiting for instructions to fetch/decode
- 🟠 **Backend Bound** (`#f39c12`) - CPU stalled on execution resources or memory

**L2 Categories:**
- Light/Heavy Operations, Branch Mispredicts, Machine Clears, Fetch Latency/Bandwidth, Memory Bound, Core Bound

**L3 Categories:**
- Detailed breakdowns including FP/Int arithmetic, memory hierarchy (L1/L2/L3/DRAM), port utilization, etc.

---

## sum1d
**Input:** `n=30,000,000` (sum integers from 0 to n)

```python
def sum1d(s, e):
    """Sum integers from s to e"""
    c = 0
    for i in range(s, e):
        c += i
    return c
```

![sum1d TMA metrics](plots/sum1d.svg)

## sum2d
**Input:** `n=2,000` (nested loop sum)

```python
def sum2d(s, e):
    """Nested loop sum"""
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c
```

![sum2d TMA metrics](plots/sum2d.svg)

## while_count
**Input:** `n=30,000,000` (while loop sum)

```python
def while_count(s, e):
    """While loop sum"""
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c
```

![while_count TMA metrics](plots/while_count.svg)

## andor
**Input:** `n=5,000,000` (boolean logic on arrays)

```python
def andor(x_arr, y_arr):
    """Boolean logic test on arrays"""
    result = 0
    for i in range(x_arr.shape[0]):
        if (x_arr[i] > 0 and x_arr[i] < 10) or (y_arr[i] > 0 and y_arr[i] < 10):
            result += 1
    return result
```

![andor TMA metrics](plots/andor.svg)

## copy_arrays
**Input:** `n=5,000,000` (1D array copy)

```python
def copy_arrays(a, b):
    """Copy 1D array elements"""
    for i in range(a.shape[0]):
        b[i] = a[i]
```

![copy_arrays TMA metrics](plots/copy_arrays.svg)

## copy_arrays2d
**Input:** `n=2,000` (2D array `2000×2000`)

```python
def copy_arrays2d(a, b):
    """Copy 2D array elements"""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]
```

![copy_arrays2d TMA metrics](plots/copy_arrays2d.svg)

## blackscholes_cnd
**Input:** `n=5,000,000` (Black-Scholes cumulative normal distribution on array)

```python
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
```

![blackscholes_cnd TMA metrics](plots/blackscholes_cnd.svg)

## matmul_naive
**Input:** `n=200` (naive matrix multiply `200×200`)

```python
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
```

![matmul_naive TMA metrics](plots/matmul_naive.svg)

## matmul_blocked
**Input:** `n=200`, `block_size=32` (blocked/tiled matrix multiply for cache optimization)

```python
def matmul_blocked(A, B, C, block_size=32):
    """Blocked/tiled matrix multiplication: C = A @ B"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    for ii in range(0, n, block_size):
        for jj in range(0, m, block_size):
            for kk in range(0, k, block_size):
                i_end = min(ii + block_size, n)
                j_end = min(jj + block_size, m)
                k_end = min(kk + block_size, k)
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        c = C[i, j]
                        for l in range(kk, k_end):
                            c += A[i, l] * B[l, j]
                        C[i, j] = c
```

![matmul_blocked TMA metrics](plots/matmul_blocked.svg)

## matmul_transpose
**Input:** `n=200` (matrix multiply with pre-transposed B for better cache locality)

```python
def matmul_transpose(A, B_T, C):
    """Matrix multiplication with pre-transposed B: C = A @ B_T.T"""
    n = A.shape[0]
    m = B_T.shape[0]
    k = A.shape[1]
    for i in range(n):
        for j in range(m):
            c = 0.0
            for l in range(k):
                c += A[i, l] * B_T[j, l]  # B_T[j, l] == original B[l, j]
            C[i, j] = c
```

![matmul_transpose TMA metrics](plots/matmul_transpose.svg)
