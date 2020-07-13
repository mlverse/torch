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
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, 3, 1)
    self$conv2 <- nn_conv2d(32, 64, 3, 1)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- nnf_relu(x)
    x <- self$conv2(x)
    x <- nnf_relu(x)
    x <- nnf_max_pool2d(x, 2)
    x <- self$dropout1(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout2(x)
    x <- self$fc2(x)
    output <- nnf_log_softmax(x, dim=1)
    output
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

