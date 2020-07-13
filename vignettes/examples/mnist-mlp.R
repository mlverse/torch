dir <- "~/Downloads/mnist"

ds <- mnist_dataset(
  dir, 
  download = TRUE, 
  transform = function(x) {
    x$to(dtype = torch_float())/256
  }
)
dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)

net <- nn_module(
  "Net",
  initialize = function() {
    self$fc1 <- nn_linear(784, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x %>% 
      torch_flatten(start_dim = 2) %>% 
      self$fc1() %>% 
      nnf_relu() %>% 
      self$fc2() %>% 
      nnf_log_softmax(dim = 1)
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

