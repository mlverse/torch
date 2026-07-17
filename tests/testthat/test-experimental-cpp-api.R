test_that("the experimental C++ Tensor API works through Lantern", {
  source_cpp_test("experimental_tensor.cpp")

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
  expect_equal(
    as.numeric(cpp_experimental_tensor_scalar_overloads(torch_tensor(c(1, 2)))),
    c(81, 144)
  )
  expect_equal(
    as_array(cpp_experimental_tensor_inplace_overloads(
      torch_zeros(2, 2), torch_ones(2, 2)
    )),
    matrix(c(7, 7, 1, 1), nrow = 2, byrow = TRUE)
  )
  expect_equal(
    as.numeric(cpp_experimental_tensor_cat_overloads(
      torch_tensor(c(1, 2)), torch_tensor(c(3, 4))
    )),
    c(1, 2, 3, 4)
  )

  expect_equal(as.numeric(cpp_experimental_creation_zeros()), rep(0, 6))
  expect_equal(as.numeric(cpp_experimental_creation_arange()), c(1, 3, 5, 7))
  expect_equal(
    as.numeric(cpp_experimental_creation_eye()),
    c(1, 0, 0, 0, 1, 0)
  )
  expect_equal(
    as.numeric(cpp_experimental_creation_full_like(x)),
    rep(7, 4)
  )
  expect_equal(cpp_experimental_creation_random_sizes(), c(2, 3))
  options_tensor <- zeros(5, 3)
  expect_equal(as.numeric(options_tensor), rep(0, 15))
  expect_true(options_tensor$dtype == torch_float64())

  layer <- torch_tensor(matrix(as.numeric(1:6), nrow = 3, ncol = 2))
  coefficients <- torch_tensor(matrix(c(10, 20, 30, 40), nrow = 2))
  intercept <- torch_tensor(matrix(c(1, 0), nrow = 1))
  expect_equal(
    as_array(cpp_experimental_nn2poly_linear(layer, coefficients)),
    as_array(layer$t()$matmul(torch_cat(list(intercept, coefficients))))
  )

  matrix <- torch_zeros(3, 2)
  values <- torch_tensor(c(1, 2, 3))
  expect_equal(
    as_array(cpp_experimental_nn2poly_add_partition(matrix, 1, 2, values)),
    cbind(c(0, 0, 0), c(2, 4, 6))
  )

  input <- torch_tensor(cbind(c(1, 2, 3), c(4, 5, 6)))
  expect_equal(
    as_array(cpp_experimental_nn2poly_add_poly_eval(
      torch_zeros(3, 2), 0, input, 1, c(2, 3, 4)
    )),
    cbind(2 + 3 * c(4, 5, 6) + 4 * c(4, 5, 6)^2, c(0, 0, 0))
  )

  partition_input <- torch_tensor(
    cbind(c(2, 3, 4), c(5, 6, 7), c(2, 3, 4))
  )
  expect_equal(
    as_array(cpp_experimental_nn2poly_accumulate_partition(
      partition_input, 1, c(1, 2), c(2, 3), torch_zeros(3, 2)
    )),
    cbind(c(0, 0, 0), c(2, 3, 4)^2 * c(5, 6, 7)^2 * c(2, 3, 4)^3)
  )
})
