# Compare installed torch builds with: Rscript bench/bench-optimizer.R
# Many small parameters expose per-parameter optimizer overhead. Set
# TORCH_BENCH_OPTIMIZER to adamw, adagrad, rmsprop, adadelta, rprop, or sgd
# to benchmark another optimizer. The SGD variant uses momentum.
library(torch)

parameters <- lapply(seq_len(250), function(i) {
  p <- torch_tensor(1, requires_grad = TRUE)
  p$grad <- torch_tensor(1)
  p
})
optimizer_name <- Sys.getenv("TORCH_BENCH_OPTIMIZER", "adam")
optimizer_fn <- switch(optimizer_name, adam = optim_adam, adamw = optim_adamw,
                       adagrad = optim_adagrad, rmsprop = optim_rmsprop,
                       adadelta = optim_adadelta, rprop = optim_rprop,
                       sgd = function(params) optim_sgd(params, lr = 0.01, momentum = 0.9),
                       stop("Unknown optimizer: ", optimizer_name))
optimizer <- optimizer_fn(parameters)
optimizer$step() # Initialize optimizer state before timing.

cat("torch:", as.character(packageVersion("torch")), "optimizer:", optimizer_name, "\n")
elapsed <- numeric(6)
for (i in seq_along(elapsed)) {
  gc()
  elapsed[i] <- system.time(for (step in seq_len(5)) optimizer$step())[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f\n", i, elapsed[i]))
}
cat(sprintf("median elapsed=%.3f seconds\n", median(elapsed)))
