# Compare installed torch builds with: Rscript bench/bench-gradient-access.R
library(torch)

x <- torch_tensor(1, requires_grad = TRUE)
x$grad <- torch_ones(1)
grad <- x$grad

# Warm up the R compiler and native calls.
for (i in seq_len(1000)) {
  invisible(x$grad)
  invisible(torch:::is_undefined_tensor(grad))
}

for (i in seq_len(3)) {
  grad_time <- system.time(for (j in seq_len(100000)) invisible(x$grad))[["elapsed"]]
  check_time <- system.time(for (j in seq_len(100000)) {
    invisible(torch:::is_undefined_tensor(grad))
  })[["elapsed"]]
  cat(sprintf("trial=%d gradient=%.3f undefined_check=%.3f seconds\n", i, grad_time, check_time))
}
