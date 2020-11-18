library(torch)
library(torchvision)

ds <- tiny_imagenet_dataset(root = "~/Downloads/tiny-imagenet")
dl <- dataloader(ds, batch_size = 32)

system.time({
  n <- 1
  for (batch in enumerate(dl)) {
    x <- batch$x$shape
    y <- batch$y$shape
    n <- n + 1
    if (n > 100)
      break
  }
})

plan(multisession(workers = 32))

system.time({
  n <- 1
  for (batch in enumerate(dl)) {
    x <- batch$x$shape
    y <- batch$y$shape
    n <- n + 1
    if (n > 10)
      break
  }
})

