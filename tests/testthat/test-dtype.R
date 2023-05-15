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

test_that("can set select devices using strings", {
  dtypes <- list(
    "float32" = torch_float32(),
    "float" = torch_float(),
    "float64" = torch_float64(),
    "double" = torch_double(),
    "float16" = torch_float16(),
    "half" = torch_half(),
    "uint8" = torch_uint8(),
    "int8" = torch_int8(),
    "int16" = torch_int16(),
    "short" = torch_short(),
    "int32" = torch_int32(),
    "int" = torch_int(),
    "int64" = torch_int64(),
    "long" = torch_long(),
    "bool" = torch_bool()
  )
  
  for(i in seq_along(dtypes)) {
    x <- torch_empty(10, 10, dtype = names(dtypes)[i])
    y <- torch_empty(10, 10, dtype = dtypes[[i]])
    
    expect_true(x$device == y$device)  
  }
  
})
