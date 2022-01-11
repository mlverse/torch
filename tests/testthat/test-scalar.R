context("scalar")

test_that("double scalar", {
  expect_true(is_torch_scalar(torch_scalar(1)))
})

test_that("int scalar", {
  expect_true(is_torch_scalar(torch_scalar(1L)))
})

test_that("bool scalar", {
  expect_true(is_torch_scalar(torch_scalar(TRUE)))
})
