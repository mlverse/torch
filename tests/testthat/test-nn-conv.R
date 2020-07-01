test_that("conv3d", {
  
  input <- torch_randn(20, 16, 10, 50, 100)
  
  m <- nn_conv3d(16, 33, 3, stride=2)
  o <- m(input)
  
  expect_tensor_shape(o, c(20, 33,  4, 24, 49))
  
  m <- nn_conv3d(16, 33, c(3, 5, 2), stride=c(2, 1, 1), padding=c(4, 2, 0))
  o <- m(input)
  
  expect_tensor_shape(o, c(20, 33,  8, 50, 99))
})

test_that("nn_conv_transpose1d", {
  
  m <- nn_conv_transpose1d(32, 16, 2)
  input <- torch_randn(10, 32, 2)
  output <- m(input)
  
  expect_tensor_shape(output, c(10, 16, 3))
  
})