#!/usr/bin/env Rscript
# Benchmark: POSIX shared memory IPC vs default callr pipe for dataloaders.
# Measures only data transfer time (excludes worker startup).

library(torch)

make_ds <- function(n, p) {
  dataset(
    initialize = function() {
      self$x <- matrix(rnorm(n * p), nrow = n, ncol = p)
    },
    .getitem = function(i) {
      torch_tensor(self$x[i, ])
    },
    .length = function() { nrow(self$x) }
  )
}

bench_transfer <- function(n, p, bs, nw, n_reps = 3) {
  times <- numeric(n_reps)
  for (r in seq_len(n_reps)) {
    dl <- dataloader(make_ds(n, p)(), batch_size = bs, num_workers = nw)
    iter <- dataloader_make_iter(dl)
    # first batch warms up workers, discard it
    dataloader_next(iter)
    start <- proc.time()["elapsed"]
    while (!is.null(dataloader_next(iter, completed = NULL))) { }
    times[r] <- proc.time()["elapsed"] - start
  }
  median(times)
}

configs <- list(
  list(n = 500,  p = 1000,    bs = 32, label = "500x1K,    bs=32"),
  list(n = 200,  p = 50000,   bs = 64, label = "200x50K,   bs=64"),
  list(n = 200,  p = 100000,  bs = 64, label = "200x100K,  bs=64"),
  list(n = 100,  p = 500000,  bs = 64, label = "100x500K,  bs=64"),
  list(n = 100,  p = 1000000, bs = 32, label = "100x1M,    bs=32")
)

cat("| Config | Default | SHM | Speedup | MB/batch |\n")
cat("|---|---|---|---|---|\n")

for (cfg in configs) {
  batch_mb <- cfg$bs * cfg$p * 4 / 1024^2

  options(torch.dataloader_use_mori = FALSE)
  t_default <- bench_transfer(cfg$n, cfg$p, cfg$bs, 2)

  options(torch.dataloader_use_mori = TRUE)
  t_shm <- bench_transfer(cfg$n, cfg$p, cfg$bs, 2)

  cat(sprintf("| %s | %.3fs | %.3fs | %.2fx | %.1f |\n",
      cfg$label, t_default, t_shm, t_default / t_shm, batch_mb))
}
