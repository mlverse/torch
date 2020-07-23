test_that("save tensor", {
  fname <- tempfile(fileext = "pt")
  x <- torch_randn(10, 10)
  torch_save(x, fname)
  y <- torch_load(fname)
  
  expect_equal_to_tensor(x, y)
})

test_that("save a module", {
  
  fname <- tempfile(fileext = "pt")
  
  Net <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 1)
      self$norm <- nn_batch_norm1d(1)
    },
    forward = function(x) {
      x <- self$linear(x)
      x <- self$norm(x)
      x
    }
  )
  net <- Net()
  
  torch_save(net, fname)
  reloaded_net <- torch_load(fname)
  
  x <- torch_randn(100, 10)
  expect_equal_to_tensor(net(x), reloaded_net(x))
  
})

test_that("save more complicated module", {
  
  Net <- nn_module(
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
  fname <- tempfile(fileext = ".pt")
  
  net <- Net()
  

  torch_save(net, fname)
  reloaded_net <- torch_load(fname)
  
  expect_equal_to_tensor(net$conv1$parameters$weight, 
                         reloaded_net$conv1$parameters$weight) 
  expect_equal_to_tensor(net$conv1$parameters$bias, 
                         reloaded_net$conv1$parameters$bias) 
  
  expect_equal_to_tensor(net$conv2$parameters$weight, 
                         reloaded_net$conv2$parameters$weight) 
  expect_equal_to_tensor(net$conv2$parameters$bias, 
                         reloaded_net$conv2$parameters$bias) 
  
  expect_equal_to_tensor(net$fc1$parameters$weight, 
                         reloaded_net$fc1$parameters$weight) 
  expect_equal_to_tensor(net$fc1$parameters$bias, 
                         reloaded_net$fc1$parameters$bias) 
  
  expect_equal_to_tensor(net$fc2$parameters$weight, 
                         reloaded_net$fc2$parameters$weight) 
  expect_equal_to_tensor(net$fc2$parameters$bias, 
                         reloaded_net$fc2$parameters$bias) 
  
  net$train(FALSE)
  reloaded_net$train(FALSE)
  
  x <- torch_randn(10, 1, 28, 28)
  expect_equal_to_tensor(net(x), reloaded_net(x))
  
})

test_that("save alexnet like model", {
  
  net <- nn_module(
    "Net",
    initialize = function() {
      self$features <- nn_sequential(
        nn_conv2d(3, 5, kernel_size = 11, stride = 4, padding = 2),
        nn_relu()
      )
      self$avgpool <- nn_max_pool2d(c(6,6))
      self$classifier <- nn_sequential(
        nn_dropout(),
        nn_linear(10, 10),
        nn_relu(),
        nn_dropout()
      )
    },
    forward = function(x) {
      x <- self$features(x)
      x <- self$avgpool(x)
      x <- torch_flatten(x, start_dim = 2)
      x <- self$classifier(x)
    }
  )
  
  model <- net()
  
  fname <- tempfile(fileext = ".pt")
  torch_save(model, fname)
  m <- torch_load(fname)
  
  pars <- model$parameters
  r_pars <- m$parameters
  
  for (i in seq_along(pars))
    expect_equal_to_tensor(pars[[i]], r_pars[[i]])
  
})

