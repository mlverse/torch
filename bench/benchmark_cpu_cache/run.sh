#!/bin/bash
# Benchmark CPU block cache: enabled vs disabled
# Usage: bash bench/benchmark_cpu_cache/run.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LATENT="${LATENT:-500}"

echo "============================================"
echo "1/3  R torch (cache enabled)"
echo "============================================"
Rscript "$DIR/benchmark.R" --latent="$LATENT"

echo ""
echo "============================================"
echo "2/3  Cache disabled"
echo "============================================"
Rscript "$DIR/benchmark.R" --no-cache --latent="$LATENT"

echo ""
echo "============================================"
echo "3/3  Python (PyTorch 2.8.0)"
echo "============================================"
uv run --with torch==2.8.0 --with numpy python "$DIR/benchmark.py" --latent="$LATENT"
