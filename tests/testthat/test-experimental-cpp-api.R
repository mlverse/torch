test_that("the experimental C++ Tensor API works through Lantern", {
  Rcpp::sourceCpp(test_path("..", "cpp", "experimental_tensor.cpp"))

  x <- torch_tensor(matrix(c(-1, 2, 3, -4), nrow = 2))

  expect_equal(cpp_experimental_tensor_sizes(x), c(2, 2))
  expect_equal(
    as.numeric(cpp_experimental_tensor_relu_reshape(x)),
    c(0, 3, 2, 0)
  )
  expect_equal(
    as.numeric(cpp_experimental_tensor_generated_methods(x)),
    sin(c(1, 4, 9, 16)),
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(cpp_experimental_tensor_native_integer_argument(x)),
    c(-1, 3)
  )
  expect_true(cpp_experimental_tensor_native_bool_return(x))
})
