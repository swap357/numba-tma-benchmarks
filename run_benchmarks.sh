#!/bin/bash
# Run all use case benchmarks and compare Python vs JIT

echo "Running all use case benchmarks..."
echo ""

functions=("sum1d" "sum2d" "while_count" "andor" "copy_arrays" "copy_arrays2d" "blackscholes_cnd")
configs=(
    "sum1d:1000000"
    "sum2d:1000"
    "while_count:1000000"
    "andor:1000"
    "copy_arrays:1000000"
    "copy_arrays2d:1000"
    "blackscholes_cnd:10000"
)

for config in "${configs[@]}"; do
    IFS=':' read -r func n <<< "$config"
    echo "=== $func (n=$n) ==="
    echo -n "Python: "
    python benchmark_usecases.py -f "$func" -e py -n "$n" --reps 5
    echo -n "JIT:    "
    python benchmark_usecases.py -f "$func" -e jit -n "$n" --reps 5
    echo ""
done

