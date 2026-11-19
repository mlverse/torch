# Compare installed torch builds with: Rscript bench/bench-conv-dispatch.R
# A small convolution makes dispatch overhead visible.
library(torch)

torch_set_num_threads(1)
input <- torch_ones(1, 1, 3, 3)
weight <- torch_ones(1, 1, 1, 1)
for (i in seq_len(100)) invisible(torch_conv2d(input, weight))

for (i in seq_len(3)) {
  elapsed <- system.time(for (j in seq_len(10000)) {
    invisible(torch_conv2d(input, weight))
  })[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f seconds\n", i, elapsed))
}
