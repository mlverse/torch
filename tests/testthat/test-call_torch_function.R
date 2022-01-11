test_that("call_torch_function works as expected", {
  x <- torch_randn(10, requires_grad = TRUE)
  y <- torch_tanh(x)
  y$backward(torch_ones_like(x))
  expect_warning(
    x_grad <- call_torch_function("torch_tanh_backward", 1, y),
    "allows access to unexported functions, please use with caution"
  )
  expect_tensor(x_grad)
  expect_equal_to_tensor(x$grad, x_grad)
  x_grad2 <- call_torch_function("torch_tanh_backward", !!!list(2, y), quiet = TRUE)
  expect_tensor(x_grad2)
  x_grad3 <- call_torch_function("torch_tanh_backward", !!!list(output = y, grad_output = 2), quiet = TRUE)
  expect_tensor(x_grad3)
  expect_equal_to_tensor(x_grad2, x_grad3)

  expect_error(
    call_torch_function("torsh_tanh_backward", 1, y, quiet = TRUE),
    "torch_"
  )

  expect_error(
    call_torch_function("torch_tank_backward", 1, y, quiet = TRUE),
    "check your spelling"
  )
})
