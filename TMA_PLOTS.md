# TMA Metrics Visualization

Top-down Microarchitecture Analysis (TMA) hierarchy for all benchmark functions, showing absolute CPU cycles at 4.0 GHz.

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

## sum1d
![sum1d TMA metrics](plots/sum1d.svg)

## sum2d
![sum2d TMA metrics](plots/sum2d.svg)

## while_count
![while_count TMA metrics](plots/while_count.svg)

## andor
![andor TMA metrics](plots/andor.svg)

## copy_arrays
![copy_arrays TMA metrics](plots/copy_arrays.svg)

## copy_arrays2d
![copy_arrays2d TMA metrics](plots/copy_arrays2d.svg)

## blackscholes_cnd
![blackscholes_cnd TMA metrics](plots/blackscholes_cnd.svg)

