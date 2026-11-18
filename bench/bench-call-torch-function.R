# Compare installed torch builds with: Rscript bench/bench-call-torch-function.R
library(torch)

x <- torch_tensor(1)
for (i in seq_len(20)) invisible(call_torch_function("torch_relu", x, quiet = TRUE))
for (i in seq_len(3)) {
  elapsed <- system.time(for (j in seq_len(500)) {
    invisible(call_torch_function("torch_relu", x, quiet = TRUE))
  })[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f seconds\n", i, elapsed))
}
