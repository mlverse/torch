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

test_that("nn_conv_transpose2d", {
  
  input <- torch_randn(20, 16, 50, 100)
  m <- nn_conv_transpose2d(16, 33, 3, stride=2)
  
  output <- m(input)
  expect_tensor_shape(output, c(20, 33, 101, 201))
  
  m <- nn_conv_transpose2d(16, 33, c(3, 5), stride=c(2, 1), padding=c(4, 2))
  output <- m(input)
  expect_tensor_shape(output, c(20, 33, 93, 100))
  
  # exact output size can be also specified as an argument
  input <- torch_randn(1, 16, 12, 12)
  downsample <- nn_conv2d(16, 16, 3, stride=2, padding=1)
  upsample <- nn_conv_transpose2d(16, 16, 3, stride=2, padding=1)
  h <- downsample(input)
  h$size()
  output <- upsample(h, output_size=input$size())
  
  expect_equal(output$size(), input$size())
})

test_that("nn_conv_transpose3d", {
  
  input <- torch_randn(20, 16, 10, 50, 100)
  
  # With square kernels and equal stride
  m <- nn_conv_transpose3d(16, 33, 3, stride=2)
  output <- m(input)
  expect_tensor_shape(output, c(20L, 33L, 21L, 101L, 201L))
  
  # non-square kernels and unequal stride and with padding
  m <- nn_conv_transpose3d(16, 33, c(3, 5, 2), stride=c(2, 1, 1), padding=c(0, 4, 2))
  output <- m(input)
  
  expect_tensor_shape(output, c(20L, 33L, 21L, 46L, 97L))
})