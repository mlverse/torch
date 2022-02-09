context("gen-method")

test_that("__and__", {
  x <- torch_tensor(TRUE)
  y <- x$`__and__`(x)
  expect_tensor(y)
  expect_equal_to_tensor(y, x)

  x <- torch_tensor(c(TRUE, FALSE))
  y <- x$`__and__`(TRUE)
  expect_tensor(y)
  expect_equal_to_tensor(y, x)
})

test_that("add", {
  x <- torch_tensor(1L, dtype = torch_long())
  expect_equal_to_r(x$add(1L)$to(dtype = torch_int()), 2L)

  x <- torch_tensor(1L, dtype = torch_long())
  expect_error(y <- x$add(1L), regexp = NA)
  expect_true(x$dtype == torch_long())

  x <- torch_tensor(1)
  expect_equal_to_r(x$add(1), 2)
})

test_that("clamp", {
  x <- torch_randn(5)
  expect_error(x$clamp(1), regexp = NA)
})

test_that("clone", {
  x <- torch_randn(10, 10)
  y <- x$clone()

  expect_equal_to_tensor(x, y)
  expect_true(!x$storage()$data_ptr() == y$storage()$data_ptr())
})

test_that("item", {
  x <- torch_tensor(1)
  expect_equal(x$item(), 1)

  x <- torch_tensor(1L)
  expect_equal(x$item(), 1L)

  x <- torch_tensor(TRUE)
  expect_equal(x$item(), TRUE)

  x <- torch_tensor(1.5)
  expect_equal(x$item(), 1.5)

  x <- torch_tensor(1.5, dtype = torch_double())
  expect_equal(x$item(), 1.5)
})

test_that("new_full", {
  x <- torch_randn(2, 2)
  expect_equal_to_tensor(
    x$new_full(c(3, 3), 1),
    torch_ones(3, 3)
  )
})

test_that("permute", {
  x <- torch_randn(2, 3, 4)
  y <- x$permute(c(3, 2, 1))

  expect_tensor_shape(y, c(4, 3, 2))

  expect_error(
    x$permute(c(2, 1, 0)),
    regex = "Indexing starts at 1 but found a 0.",
    fixed = TRUE
  )
})
