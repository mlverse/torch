test_that("nn_linear", {
  
  linear <- nn_linear(10, 1)
  x <- torch_randn(10, 10)
  
  y <- linear(x)
  
  expect_tensor(y)
  expect_length(as_array(y), 10)
})
