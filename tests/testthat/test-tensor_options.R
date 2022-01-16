context("tensor-options")

test_that("tensor options works", {
  x <- torch_tensor_options()
  expect_true(is.list(x))

  options <- torch_tensor_options(dtype = torch_bool())
  expect_true(is.list(options))
})
