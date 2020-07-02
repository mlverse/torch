context("reduction")

test_that("reduction works", {
  expect_identical(torch_reduction_none(), 0)
  expect_identical(torch_reduction_mean(), 1)
  expect_identical(torch_reduction_sum(), 2)
})
