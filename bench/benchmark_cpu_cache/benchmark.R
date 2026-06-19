library(torch)

args <- commandArgs(trailingOnly = TRUE)
cache_enabled <- !("--no-cache" %in% args)
latent_arg <- grep("^--latent=", args, value = TRUE)
latent <- if (length(latent_arg)) as.integer(sub("--latent=", "", latent_arg)) else 500L

set_cpu_allocator_config(cache_enabled = cache_enabled)

cat("torch version:", as.character(packageVersion("torch")), "\n")
cat("cache:", if (cache_enabled) "enabled" else "disabled", "\n")
cat("latent:", latent, "\n")

p <- 100
steps <- 1000
n <- 1000
nreps <- 5

se <- function(x) sd(x) / sqrt(length(x))

results <- numeric(nreps)
for (i in 1:nreps) {
  X <- torch_randn(n, p)
  Y <- torch_randn(n, 1)

  net <- nn_sequential(
    nn_linear(p, latent),
    nn_relu(),
    nn_linear(latent, 1)
  )
  opt <- optim_adam(net$parameters, lr = 0.01)

  t1 <- Sys.time()
  for (j in 1:steps) {
    opt$zero_grad(set_to_none = TRUE)
    Y_hat <- net(X)
    loss <- nnf_mse_loss(Y, Y_hat)
    loss$backward()
    opt$step()
  }
  loss$item()
  results[i] <- as.numeric(Sys.time() - t1)
  cat(sprintf("Rep %d: %.2fs\n", i, results[i]))
}

cat(sprintf("\nmean=%.3fs  se=%.3fs\n", mean(results), se(results)))
