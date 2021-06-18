test_that("script module parameters", {
  
  script_module <- jit_load("assets/linear.pt")
  parameters <- script_module$parameters
  
  expect_equal(names(parameters), c("weight", "bias"))
  expect_tensor_shape(parameters$weight, c(10, 10))
  expect_tensor_shape(parameters$bias, c(10))
  
})

test_that("parameters are modifiable in-place", {
  script_module <- jit_load("assets/linear.pt")
  parameters <- script_module$parameters
  
  with_no_grad({
    parameters$weight$zero_()  
  })
  
  parameters <- script_module$parameters
  expect_equal_to_tensor(parameters$weight, torch_zeros(10, 10))
})

test_that("train works", {
  script_module <- jit_load("assets/linear.pt")
  
  script_module$train(TRUE)
  expect_true(script_module$is_training)
  
  script_module$train(FALSE)
  expect_true(!script_module$is_training)
  
  script_module$train(TRUE)
  expect_true(script_module$is_training)
})

test_that("can register parameters", {
  script_module <- jit_load("assets/linear.pt")
  x <- torch_tensor(1)
  script_module$register_parameter("hello", x)
  parameters <- script_module$parameters
  expect_equal(names(parameters), c("weight", "bias", "hello"))
})

test_that("can register buffers", {

  script_module <- jit_load("assets/linear.pt")
  buffers <- script_module$buffers
  
  expect_length(buffers, 0)
  
  script_module$register_buffer("hello", torch_tensor(1))
  buffers <- script_module$buffers
  
  expect_length(buffers, 1)
  expect_equal(names(buffers), "hello")
  expect_equal_to_tensor(buffers[[1]], torch_tensor(1))
    
  
})

test_that("can move to device", {
  
  skip_if_cuda_not_available()
  script_module <- jit_load("assets/linear.pt")
  script_module$to("cuda")
  parameters <- script_module$parameters
  
  expect_true(parameters$weight$device == torch_device("cuda"))
  expect_true(parameters$bias$device == torch_device("cuda"))
  
})


