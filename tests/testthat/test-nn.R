test_that("nn_module", {
  my_net <- nn_module(
    "my_net",
    initialize = function(n_inputs, n_outputs) {
      self$W <- nn_parameter(torch_randn(n_inputs, n_outputs))
      self$b <- nn_parameter(torch_zeros(n_outputs))
    },
    forward = function(x) {
      torch_addmm(self$b, x, self$W)
    }
  )
  
  model <- my_net(1,1)
  expect_s3_class(model, "nn_module")
  expect_s3_class(model, "my_net")
  expect_length(model$parameters, 2)
  expect_tensor(model(torch_randn(10,1)))
})

test_that("nn_modules can have child modules", {
  my_net <- nn_module(
    "my_net",
    initialize = function(n_inputs, n_outputs) {
      self$linear <- nn_linear(n_inputs, n_outputs)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  
  model <- my_net(1,2)
  x <- torch_randn(1, 1)
  output <- model(x)
  
  expect_s3_class(model, "nn_module")
  expect_s3_class(model, "my_net")
  expect_length(model$parameters, 2)
  expect_tensor(output)
  expect_equal(output$dim(), 2)
  
})

test_that("nn_sequential", {
  model <- nn_sequential(
    nn_linear(10, 100),
    nn_relu(),
    nn_linear(100, 1)
  )
  
  input <- torch_randn(1000, 10)
  output <- model(input)
  
  expect_tensor(output)
  expect_s3_class(model, "nn_sequential")
  expect_s3_class(model, "nn_module")
  expect_equal(output$shape, c(1000, 1))
  expect_length(model$parameters, 4)
  
  model <- nn_sequential(
    name = "mynet",
    nn_linear(10, 100),
    nn_relu(),
    nn_linear(100, 1)
  )
  expect_s3_class(model, "mynet")
  expect_s3_class(model, "nn_module")
})
