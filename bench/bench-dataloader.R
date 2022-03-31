devtools::load_all()

workers <- c(0, 2, 4, 8)
time_per_batch <- c(0.01, 0.1, 0.5, 1, 2, 5)
size_of_tensor <- c(1000, 10000, 100000, 250000)
batch_size <- c(32, 64, 128)
time_for_forward <- c(0.01, 0.1, 0.5, 1, 2)
grid <- expand.grid(
  workers = workers,
  time_per_batch = time_per_batch,
  size_of_tensor = size_of_tensor,
  time_for_forward = time_for_forward,
  batch_size = batch_size,
  socket = c(TRUE, FALSE)
)
grid$elapsed <- NA

bench_ds <- dataset(
  initialize = function(time_per_batch, batch_size, size_of_tensor) {
    self$time_per_batch <- time_per_batch
    self$batch_size <- batch_size
    self$size_of_tensor <- size_of_tensor
  },
  .getitem = function(i) {
    Sys.sleep(self$time_per_batch/self$batch_size)
    list(x = torch_randn(self$size_of_tensor))
  },
  .length = function() {60000}
)

# stopped with i = 1368
for (i in 1:nrow(grid)) {

  g <- as.list(grid[i,])

  # if (g$workers != 0)
  #   plan(multisession(workers = g$workers))

  ds <- bench_ds(g$time_per_batch, g$batch_size, g$size_of_tensor)
  withr::with_options(list(torch.dataloader_use_socket_con = g$socket), {
    dl <- dataloader(ds, batch_size = g$batch_size, num_workers = g$workers)  
  })
  
  time <- system.time({
    n <- 1
    coro::loop(for (batch in dl) {
      Sys.sleep(g$time_for_forward)
      x <- batch$x$shape
      y <- batch$y$shape
      n <- n + 1
      if (n > 10)
        break
    })
  })

  grid$elapsed[i] <- time["elapsed"]
}

saveRDS(grid, "~/Downloads/grid-parallel1.rds")

scales::number_bytes(100)
library(tidyverse)
grid %>%
  mutate(
    bytes = batch_size * size_of_tensor * 4,
    bytes_c = forcats::fct_reorder(
      scales::number_bytes(bytes),
      bytes
    )
  ) %>%  # float32 are 4 bytes
  ggplot(aes(x = workers, y = elapsed, colour = bytes_c)) +
  geom_point() +
  geom_line() +
  facet_wrap(~time_per_batch)


