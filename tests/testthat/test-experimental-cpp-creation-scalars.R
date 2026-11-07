test_that("C++ creation functions preserve integral scalars", {
  cpp_extension_package()
  result <- cpp_experimental_creation_integral_scalars()

  for (name in c("scalar", "literal", "full", "full_vector", "full_like",
                 "full_like_options", "fill", "arange", "mixed_integer_arange")) {
    expect_true(result[[name]], info = name)
  }

  for (name in c("integer_full", "integer_arange", "integer_arange_options",
                 "integer_arange_start", "integer_arange_step")) {
    expect_true(result[[name]]$dtype == torch_int64(), info = name)
  }
  expect_equal(as.numeric(result$integer_full), c(3, 3))
  expect_equal(as.numeric(result$integer_arange), 0:3)
  expect_equal(as.numeric(result$integer_arange_options), 0:3)
  expect_equal(as.numeric(result$integer_arange_start), 1:3)
  expect_equal(as.numeric(result$integer_arange_step), c(1, 3))

  for (name in c("floating_full", "float_full", "floating_arange",
                 "mixed_arange", "scalar_default")) {
    expect_true(result[[name]]$dtype == torch_float32(), info = name)
  }
  expect_true(result$bool_full$dtype == torch_bool())
  expect_equal(as.logical(result$bool_full), c(TRUE, TRUE))
  expect_error(cpp_experimental_creation_unsigned_overflow(),
               "scalar value exceeds the int64_t range")
  expect_equal(as.numeric(result$floating_full), c(1.5, 1.5))
  expect_equal(as.numeric(result$float_full), c(1.5, 1.5))
  expect_equal(as.numeric(result$floating_arange), c(1, 1.5, 2, 2.5))
  expect_equal(as.numeric(result$mixed_arange), c(1, 1.5, 2, 2.5))
})
