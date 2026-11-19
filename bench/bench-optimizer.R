# Compare installed torch builds with: Rscript bench/bench-optimizer.R
# Many small parameters expose per-parameter optimizer overhead.
library(torch)

parameters <- lapply(seq_len(250), function(i) {
  p <- torch_tensor(1, requires_grad = TRUE)
  p$grad <- torch_tensor(1)
  p
})
optimizer <- optim_adam(parameters)
optimizer$step() # Initialize optimizer state before timing.

cat("torch:", as.character(packageVersion("torch")), "\n")
elapsed <- numeric(6)
for (i in seq_along(elapsed)) {
  gc()
  elapsed[i] <- system.time(for (step in seq_len(5)) optimizer$step())[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f\n", i, elapsed[i]))
}
cat(sprintf("median elapsed=%.3f seconds\n", median(elapsed)))
