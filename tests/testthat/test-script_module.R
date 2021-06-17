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
