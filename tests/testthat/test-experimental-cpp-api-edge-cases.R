test_that("scalar and zero-sized tensors cross the package boundary", {
  cpp_extension_package()

  scalar <- cpp_experimental_edge_scalar_tensor(3.5)
  scalar_metadata <- cpp_experimental_tensor_metadata(scalar)
  expect_equal(scalar_metadata$dim, 0)
  expect_equal(scalar_metadata$numel, 1)
  expect_equal(scalar_metadata$sizes, integer())
  expect_equal(as.numeric(scalar), 3.5)

  empty <- cpp_experimental_edge_zero_sized()
  empty_metadata <- cpp_experimental_tensor_metadata(empty)
  expect_equal(empty_metadata$sizes, c(0, 3))
  expect_equal(empty_metadata$numel, 0)
  expect_length(as.numeric(empty), 0)
})

test_that("negative indices and non-contiguous tensors behave like LibTorch", {
  cpp_extension_package()
  x <- torch_tensor(matrix(as.numeric(1:6), nrow = 2))

  selected <- cpp_experimental_edge_negative_indices(x)
  expect_equal(as_array(selected), matrix(c(5, 6), ncol = 1))
  expect_false(cpp_experimental_edge_transpose_is_contiguous(x))

  contiguous <- cpp_experimental_edge_make_contiguous(x)
  expect_true(contiguous$is_contiguous())
  expect_equal(as_array(contiguous), as_array(x$t()))
})

test_that("clone ownership and view aliasing are preserved", {
  cpp_extension_package()
  x <- torch_tensor(matrix(as.numeric(1:4), nrow = 2))

  isolated <- cpp_experimental_edge_clone_isolation(x)
  expect_equal(as.numeric(isolated), c(1, 3, 2, 4, rep(0, 4)))
  expect_equal(as.numeric(x), as.numeric(torch_tensor(matrix(as.numeric(1:4), nrow = 2))))

  aliased <- cpp_experimental_edge_view_aliasing(x)
  expected <- as_array(x)
  expected[1, ] <- 9
  expect_equal(as_array(aliased), expected)
})

test_that("temporaries, special values, and integer dtypes are safe", {
  cpp_extension_package()

  special <- torch_tensor(c(NA_real_, Inf, -Inf, -2))
  result <- cpp_experimental_edge_temporary_lifetime(special)
  expect_true(is.nan(as.numeric(result)[1]))
  expect_true(is.infinite(as.numeric(result)[2]))
  expect_true(is.infinite(as.numeric(result)[3]))
  expect_equal(as.numeric(result)[4], 9)

  integer <- torch_tensor(c(1L, 2L, 3L), dtype = torch_int64())
  integer_result <- cpp_experimental_tensor_integral_scalar_overloads(integer)
  expect_true(integer_result$dtype == torch_int64())
  expect_equal(as.numeric(integer_result), c(81, 144, 225))
})

test_that("creation options and Lantern errors propagate", {
  cpp_extension_package()

  with_grad <- cpp_experimental_edge_requires_grad_creation()
  expect_true(with_grad$requires_grad)
  expect_equal(as.numeric(with_grad), rep(1, 4))

  expect_error(
    cpp_experimental_edge_invalid_reshape(torch_ones(3)),
    "shape.*invalid|invalid.*shape|size"
  )
})
