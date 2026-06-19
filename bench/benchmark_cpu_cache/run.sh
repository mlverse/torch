#!/bin/bash
# Benchmark CPU block cache: enabled vs disabled, plus Python comparison
# Usage: bash bench/benchmark_cpu_cache/run.sh
# Override latent sizes: LATENT_SIZES="500 2000 5000" bash bench/benchmark_cpu_cache/run.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LATENT_SIZES="${LATENT_SIZES:-500 2000 5000}"

for LATENT in $LATENT_SIZES; do
  echo ""
  echo "============================================"
  echo "  latent=$LATENT"
  echo "============================================"

  echo ""
  echo "--- R torch (cache enabled) ---"
  Rscript "$DIR/benchmark.R" --latent="$LATENT"

  echo ""
  echo "--- R torch (cache disabled) ---"
  Rscript "$DIR/benchmark.R" --no-cache --latent="$LATENT"

  echo ""
  echo "--- Python PyTorch ---"
  uv run --with torch==2.8.0 --with numpy python "$DIR/benchmark.py" --latent="$LATENT"
done
