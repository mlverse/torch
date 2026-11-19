# Compare installed torch builds with: Rscript bench/bench-tensor-metadata.R
library(torch)

x <- torch_randn(2, 2)
for (i in seq_len(1000)) {
  invisible(x$shape)
  invisible(x$numel())
  invisible(x$device)
}

for (i in seq_len(3)) {
  shape <- system.time(for (j in seq_len(40000)) invisible(x$shape))[["elapsed"]]
  numel <- system.time(for (j in seq_len(40000)) invisible(x$numel()))[["elapsed"]]
  device <- system.time(for (j in seq_len(40000)) invisible(x$device))[["elapsed"]]
  cat(sprintf("trial=%d shape=%.3f numel=%.3f device=%.3f seconds\n", i, shape, numel, device))
}
