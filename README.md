# Numba TMA Benchmarks

Benchmark Numba JIT vs Python with Top-down Microarchitecture Analysis (TMA).

**Setup:**
```bash
conda env create -f environment.yml
conda activate numba-tma
```

**Run benchmarks:**
```bash
./run_benchmarks.sh
python benchmark_usecases.py -f sum1d -e jit -n 1000000 --reps 5
```

**Collect TMA metrics** (MUST use single P-core for accurate L1 decomposition):
```bash
python collect_tma.py results.csv --cpu-pin 2 --reps 5
```

**Visualize TMA breakdown:**
```bash
python plot.py results.csv              # All functions L1 overview
python plot.py results.csv sum1d        # Full hierarchy (L1+L2+L3)

# TMA percentages (relative breakdown)
python plot.py results_tma.csv sum1d

# Absolute CPU cycles (shows speedup)
python plot_cycles.py results_tma.csv sum1d
```

