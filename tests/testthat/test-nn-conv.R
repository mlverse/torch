test_that("conv3d", {
  input <- torch_randn(20, 16, 10, 5, 10)

  m <- nn_conv3d(16, 33, 3, stride = 2)
  o <- m(input)

  expect_tensor_shape(o, c(20, 33, 4, 2, 4))

  m <- nn_conv3d(16, 33, c(3, 5, 2), stride = c(2, 1, 1), padding = c(4, 2, 0))
  o <- m(input)

  expect_tensor_shape(o, c(20, 33, 8, 5, 9))
})

test_that("conv2d", {
  input <- torch_randn(1, 3, 13, 13)
  conv <- nn_conv2d(3, 10, kernel_size = 3, padding = "same")
  out <- conv(input)

  expect_tensor_shape(out, c(1, 10, 13, 13))
})

test_that("convolution fast paths match the general dispatcher", {
  convolutions <- list(torch_conv1d, torch_conv2d, torch_conv3d)
  dispatchers <- list(cpp_torch_dispatch_namespace_conv1d,
                      cpp_torch_dispatch_namespace_conv2d,
                      cpp_torch_dispatch_namespace_conv3d)
  for (n in seq_along(convolutions)) {
    input <- torch_randn(c(1, 1, rep(4, n)), requires_grad = TRUE)
    weight <- torch_randn(c(1, 1, rep(3, n)), requires_grad = TRUE)
    reference <- local({
      dispatch <- dispatchers[[n]]
      function(input, weight, bias = list(), stride = 1L, padding = 0L,
               dilation = 1L, groups = 1L) {
        dispatch(mget(x = c("input", "weight", "bias", "stride", "padding", "dilation", "groups")))
      }
    })
    for (padding in list(0L, 1L, "valid", "same")) {
      actual <- convolutions[[n]](input, weight, padding = padding)
      expected <- reference(input, weight, padding = padding)
      expect_equal_to_tensor(actual, expected)
    }

    actual <- convolutions[[n]](input, weight, padding = "same")
    actual$sum()$backward()
    input_grad <- input$grad$clone()
    weight_grad <- weight$grad$clone()
    input$grad$zero_()
    weight$grad$zero_()
    expected <- reference(input, weight, padding = "same")
    expected$sum()$backward()
    expect_equal_to_tensor(input$grad, input_grad)
    expect_equal_to_tensor(weight$grad, weight_grad)

    # Non-tensor inputs and logical padding retain the fallback.
    input_array <- as.array(input$detach())
    expect_equal_to_tensor(convolutions[[n]](input_array, weight),
                           reference(input_array, weight))
    expect_equal_to_tensor(convolutions[[n]](input, weight, padding = TRUE),
                           reference(input, weight, padding = TRUE))
    expect_error(convolutions[[n]](input, weight, padding = new.env()),
                 "Can't convert argument:padding")
    expect_error(convolutions[[n]](input), "No matching overload")
  }
})

test_that("nn_conv_transpose1d", {
  m <- nn_conv_transpose1d(32, 16, 2)
  input <- torch_randn(10, 32, 2)
  output <- m(input)

  expect_tensor_shape(output, c(10, 16, 3))
})

test_that("nn_conv_transpose2d", {
  input <- torch_randn(20, 16, 10, 10)
  m <- nn_conv_transpose2d(16, 33, 3, stride = 2)

  output <- m(input)
  expect_tensor_shape(output, c(20, 33, 21, 21))

  m <- nn_conv_transpose2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2))
  output <- m(input)
  expect_tensor_shape(output, c(20, 33, 13, 10))

  # exact output size can be also specified as an argument
  input <- torch_randn(1, 16, 12, 12)
  downsample <- nn_conv2d(16, 16, 3, stride = 2, padding = 1)
  upsample <- nn_conv_transpose2d(16, 16, 3, stride = 2, padding = 1)
  h <- downsample(input)
  h$size()
  output <- upsample(h, output_size = input$size())

  expect_equal(output$size(), input$size())
})

test_that("nn_conv_transpose3d", {
  input <- torch_randn(20, 16, 10, 5, 10)

  # With square kernels and equal stride
  m <- nn_conv_transpose3d(16, 33, 3, stride = 2)
  output <- m(input)
  expect_tensor_shape(output, c(20L, 33L, 21L, 11L, 21L))

  # non-square kernels and unequal stride and with padding
  m <- nn_conv_transpose3d(16, 33, c(3, 5, 2), stride = c(2, 1, 1), padding = c(0, 4, 2))
  output <- m(input)

  expect_tensor_shape(output, c(20L, 33L, 21L, 1L, 7L))
})
