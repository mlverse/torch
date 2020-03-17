test_that("tensor options works", {
  x <- torch_tensor_options()
  expect_true(is_torch_tensor_options(x))
  expect_output(print(x))
  
  options <- torch_tensor_options(dtype = torch_bool())
  expect_true(is_torch_tensor_options(options))
  expect_output(print(options))
})


