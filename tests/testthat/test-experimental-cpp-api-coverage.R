test_that("downstream Tensor metadata and shape operations work", {
  cpp_extension_package()
  x <- torch_tensor(matrix(as.numeric(1:6), nrow = 2))

  metadata <- cpp_experimental_tensor_metadata(x)
  expect_equal(metadata$dim, 2)
  expect_equal(metadata$numel, 6)
  expect_equal(metadata$element_size, 4)
  expect_equal(metadata$sizes, c(2, 3))
  expect_true(metadata$contiguous)
  expect_false(metadata$requires_grad)
  expect_true(metadata$defined)

  expect_equal(
    as.numeric(cpp_experimental_tensor_shape_pipeline(x)),
    as.numeric(x$t()$flatten())
  )
})

test_that("downstream reductions and comparison overloads work", {
  cpp_extension_package()
  x <- torch_tensor(c(0, 2, -3, 4))

  expect_equal(
    as.numeric(cpp_experimental_tensor_reductions(x)),
    c(-3, 4, 0, 1)
  )

  first <- torch_tensor(c(1, 4, 3))
  second <- torch_tensor(c(1, 2, 5))
  expect_equal(
    as.numeric(cpp_experimental_tensor_comparisons(first, second)),
    c(1, 2, 3)
  )
  expect_equal(
    as.numeric(cpp_experimental_namespace_out_overload(first, second)),
    c(1, 4, 5)
  )
})

test_that("downstream dtype, device, and autograd state operations work", {
  cpp_extension_package()
  x <- torch_tensor(c(1, 2, 3))

  converted <- cpp_experimental_tensor_to_dtype(x)
  expect_true(converted$dtype == torch_float64())
  expect_equal(as.numeric(converted), c(1, 2, 3))

  moved <- cpp_experimental_tensor_to_device(x)
  expect_equal(moved$device$type, "cpu")

  detached <- cpp_experimental_tensor_autograd_state(x)
  expect_false(detached$requires_grad)
  expect_equal(as.numeric(detached), c(1, 2, 3))
})
