dir <- "~/Downloads/mnist"

ds <- mnist_dataset(
  dir, 
  download = TRUE, 
  transform = function(x) {
    x <- x$to(dtype = torch_float())/256
    x[newaxis,..]
  }
)
dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)

net <- nn_module(
  "Net",
  initialize = function(num_classes = 1000) {
    self$features <- nn_sequential(
      nn_conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2),
      nnf_conv2d(64, 192, kernel_size = 5, padding = 2),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2),
      nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2)
    )
    self$avgpool <- nnf_adaptive_avg_pool2d(c(6,6))
    self$classifier <- nn_sequential(
      nn_dropout(),
      nn_linear(256 * 6 * 6, 4096),
      nn_relu(),
      nn_dropout(),
      nn_linear(4096, 4096),
      nn_relu(),
      nn_linear(4096, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, 1)
    x <- self$classifier(x)
    x
  }
)

model <- net()
optimizer <- optim_sgd(model$parameters, lr = 0.01)

epochs <- 10

for (epoch in 1:10) {
  
  pb <- progress::progress_bar$new(
    total = length(dl), 
    format = "[:bar] :eta Loss: :loss"
  )
  l <- c()
  
  for (b in enumerate(dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_nll_loss(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
    pb$tick(tokens = list(loss = mean(l)))
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}

