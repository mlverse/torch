test_that("script module parameters", {
  
  script_module <- jit_load("assets/linear.pt")
  parameters <- script_module$parameters
  
  expect_equal(names(parameters), c("weight", "bias"))
  expect_tensor_shape(parameters$weight, c(10, 10))
  expect_tensor_shape(parameters$bias, c(10))
})
