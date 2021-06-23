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
