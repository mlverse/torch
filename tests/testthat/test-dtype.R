context("dtype")

test_that("Can create dtypes", {
  expect_s3_class(torch_float32(), "torch_dtype")
  expect_s3_class(torch_float(), "torch_dtype")
  expect_s3_class(torch_float64(), "torch_dtype")
  expect_s3_class(torch_double(), "torch_dtype")
  expect_s3_class(torch_float16(), "torch_dtype")
  expect_s3_class(torch_half(), "torch_dtype")
  expect_s3_class(torch_uint8(), "torch_dtype")
  expect_s3_class(torch_int8(), "torch_dtype")
  expect_s3_class(torch_int16(), "torch_dtype")
  expect_s3_class(torch_short(), "torch_dtype")
  expect_s3_class(torch_int32(), "torch_dtype")
  expect_s3_class(torch_int(), "torch_dtype")
  expect_s3_class(torch_int64(), "torch_dtype")
  expect_s3_class(torch_long(), "torch_dtype")
  expect_s3_class(torch_bool(), "torch_dtype")
})

test_that("Can compare dtypes", {
  expect_true(torch_float32() == torch_float())
  expect_false(torch_float() == torch_int())

  expect_false(torch_float32() != torch_float())
  expect_true(torch_float() != torch_int())
})

test_that("Default dtype", {
  x <- torch_randn(10)
  expect_true(x$dtype == torch_float())
  expect_true(torch_get_default_dtype() == torch_float())

  torch_set_default_dtype(torch_float64())
  expect_true(torch_get_default_dtype() == torch_float64())
  x <- torch_randn(10)
  expect_true(x$dtype == torch_float64())

  torch_set_default_dtype(torch_float())
})
