test_that("ivalue works for bool", {
  expect_equal(ivalue_test_function(TRUE), TRUE)
  expect_equal(ivalue_test_function(FALSE), FALSE)
})

test_that("ivalue works for bool list", {
  x <- sample(c(TRUE, FALSE), replace = TRUE, 10)

  expect_equal(ivalue_test_function(x), x)
})

test_that("works for int", {
  x <- 1L
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for int list", {
  x <- 1:100
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for double", {
  x <- runif(1)
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for double list", {
  x <- runif(100)
  expect_equal(ivalue_test_function(x), x)
})

test_that("tensor works", {
  x <- torch_randn(100, 100)
  expect_equal_to_tensor(ivalue_test_function(x), x)
})

test_that("tensor list works", {
  x <- list(torch_tensor(1), torch_tensor(2))
  y <- ivalue_test_function(x)

  expect_equal_to_tensor(y[[1]], x[[1]])
  expect_equal_to_tensor(y[[2]], x[[2]])
})

test_that("works for strings", {
  x <- "hello"
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for unamed lists", {
  x <- list(
    1L, 1:10, 1.1, c(2, 3.3, 4.4), torch_tensor(1),
    list(torch_tensor(1), torch_tensor(2)),
    TRUE, c(TRUE, FALSE)
  )
  x[[length(x) + 1]] <- x
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for dicts of tensors", {
  x <- list(a = torch_tensor(1), b = torch_tensor(2))
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for named list of different types", {
  x <- list(a = torch_tensor(1), b = 1)
  expect_equal(ivalue_test_function(x), x)
})

test_that("works for named tuples", {
  x <- list(a = torch_tensor(1), b = torch_tensor(2))
  class(x) <- "jit_tuple"
  expect_equal(ivalue_test_function(x), unclass(x))
})

test_that("works for tuples", {
  x <- list(torch_tensor(1), torch_tensor(2))
  class(x) <- "jit_tuple"
  expect_equal(ivalue_test_function(x), unclass(x))
})

test_that("works with NULL", {
  expect_equal(ivalue_test_function(NULL), NULL)
})

test_that("works with scalars", {
  expect_equal(ivalue_test_function(jit_scalar(1)), 1)
  expect_equal(ivalue_test_function(jit_scalar(1L)), 1L)
  expect_equal(ivalue_test_function(jit_scalar(TRUE)), TRUE)
})
