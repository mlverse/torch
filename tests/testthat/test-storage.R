context("storage")

test_that("storage", {
  x <- torch_tensor(1)
  y <- x

  expect_equal(x$storage()$data_ptr(), y$storage()$data_ptr())

  k <- x$add_(1)
  expect_equal(x$storage()$data_ptr(), k$storage()$data_ptr())
})
