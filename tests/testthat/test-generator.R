context("generator")

test_that("can create and use simple generators", {
  x <- torch_generator()
  x$set_current_seed(12345678L)
  expect_equal(x$current_seed(), bit64::as.integer64(12345678))
  
  x$set_current_seed(bit64::as.integer64(123456789101112))
  expect_equal(x$current_seed(), bit64::as.integer64(123456789101112))
})

test_that("manual_seed works", {
  
  torch_manual_seed(1L)
  a <- torch_randn(1)
  torch_manual_seed(1L)
  b <- torch_randn(1)
  
  expect_equal_to_tensor(a, b)
  
})