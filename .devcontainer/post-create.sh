#!/usr/bin/env bash
# Installs the R package from the mounted source. Object files are kept in src/, so reinstalling
# after changing R code only relinks.
set -euo pipefail
cd "$(dirname "$0")/.."
R CMD INSTALL --no-docs --no-multiarch --no-test-load .
Rscript -e 'library(torch); print(torch_randn(2, 2)); cat("libtorch", torch:::torch_version, "\n")'
python -c 'import torch; print("PyTorch", torch.__version__)'
