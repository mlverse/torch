dir <- "~/Downloads/tiny-imagenet"

ds <- tiny_imagenet_dataset(
  dir, 
  download = TRUE, 
  transform = function(x) {
    x <- magick::image_resize(x, "224x224")
    x <- as.integer(magick::image_data(x, "rgb"))  
    x <- torch_tensor(x)
    x <- x/256
    x <- x$permute(c(3, 1, 2))
  },
  target_transform = function(x) {
    x <- torch_tensor(x, dtype = torch_long())
    x$squeeze(1)
  }
)

dl <- dataloader(ds, batch_size = 128, shuffle = TRUE)

net <- nn_module(
  "Net",
  initialize = function(num_classes = 1000) {
    self$features <- nn_sequential(
      nn_conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2),
      nn_conv2d(64, 192, kernel_size = 5, padding = 2),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2),
      nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 3, stride = 2)
    )
    self$avgpool <- nn_max_pool2d(c(6,6))
    self$classifier <- nn_sequential(
      nn_dropout(),
      nn_linear(256, 4096),
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
    x <- torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
  }
)

if (cuda_is_available()) {
  device <- torch_device("cuda")
} else {
  device <- torch_device("cpu")
}
  
model <- net(num_classes = 200)
model$to(device = device)
optimizer <- optim_adam(model$parameters)
loss_fun <- nn_cross_entropy_loss()

epochs <- 10

for (epoch in 1:50) {
  
  pb <- progress::progress_bar$new(
    total = length(dl), 
    format = "[:bar] :eta Loss: :loss"
  )
  l <- c()
  
  for (b in enumerate(dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = device))
    loss <- loss_fun(output, b[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
    pb$tick(tokens = list(loss = mean(l)))
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}

