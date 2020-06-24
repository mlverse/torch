test_that("+ works", {
  x <- torch_tensor(c(1,2,3,4))
  expect_equal_to_r(x + x, 2*c(1,2,3,4))
  expect_equal_to_r(x + 2, c(1,2,3,4) + 2)
  expect_equal_to_r(2 + x, c(1,2,3,4) + 2)
})

test_that("- works", {
  x <- torch_tensor(c(1,2,3,4))
  expect_equal_to_r(x - x, rep(0, 4))
  expect_equal_to_r(x - 2, c(1,2,3,4) - 2)
  expect_equal_to_r(2 - x, 2 - c(1,2,3,4))
})

test_that("* works", {
  x <- torch_tensor(c(1,2,3,4))
  expect_equal_to_r(x*x, c(1,2,3,4)^2)
  expect_equal_to_r(x * 2, c(1,2,3,4) * 2)
  expect_equal_to_r(2 * x, c(1,2,3,4) * 2)
})

test_that("/ works", {
  x <- torch_tensor(c(1,2,3,4))
  expect_equal_to_r(x/x, rep(1, 4))
  expect_equal_to_r(x / 2, c(1,2,3,4) / 2)
  expect_equal_to_r(12 / x, 12 / c(1,2,3,4))
})

test_that("^ works", {
  x <- torch_tensor(c(1,2,3,4))
  expect_equal_to_r(x^x, c(1,2,3,4)^c(1,2,3,4))
  expect_equal_to_r(x^ 2, c(1,2,3,4) ^ 2)
  expect_equal_to_r(2 ^ x, 2 ^ c(1,2,3,4))
})

test_that("> works", {
  x <- torch_tensor(c(2,2))
  y <- torch_tensor(c(2,3))
  expect_equal_to_r(x > y, c(FALSE, FALSE))
  expect_equal_to_r(y > x, c(FALSE, TRUE))
  expect_equal_to_r(y > 2, c(FALSE, TRUE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that(">= works", {
  x <- torch_tensor(c(2,2))
  y <- torch_tensor(c(2,3))
  expect_equal_to_r(x >= y, c(TRUE, FALSE))
  expect_equal_to_r(y >= x, c(TRUE, TRUE))
  expect_equal_to_r(y >= 2, c(TRUE, TRUE))
  expect_equal_to_r(2 >= y, c(TRUE, FALSE))
})

test_that("< works", {
  x <- torch_tensor(c(2,2))
  y <- torch_tensor(c(2,3))
  expect_equal_to_r(x < y, c(FALSE, TRUE))
  expect_equal_to_r(y < x, c(FALSE, FALSE))
  expect_equal_to_r(y < 2, c(FALSE, FALSE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that("<= works", {
  x <- torch_tensor(c(2,2))
  y <- torch_tensor(c(2,3))
  expect_equal_to_r(x <= y, c(TRUE, TRUE))
  expect_equal_to_r(y <= x, c(TRUE, FALSE))
  expect_equal_to_r(y <= 2, c(TRUE, FALSE))
  expect_equal_to_r(2 <= y, c(TRUE, TRUE))
})

test_that("== works", {
  x <- torch_tensor(c(1,2))
  y <- torch_tensor(c(2,2))
  expect_equal_to_r(x == y, c(FALSE, TRUE))
  expect_equal_to_r(x == 2, c(FALSE, TRUE))
})

test_that("!= works", {
  x <- torch_tensor(c(1,2))
  y <- torch_tensor(c(2,2))
  expect_equal_to_r(x != y, c(TRUE, FALSE))
  expect_equal_to_r(x != 2, c(TRUE, FALSE))
})

test_that("dim works", {
  x <- torch_randn(c(2,2))
  expect_equal(dim(x), c(2,2))
})

test_that("length works", {
  x <- torch_randn(c(2,2))
  expect_equal(length(x), 4)
})

test_that("as.*", {
  x <- array(runif(200), dim = c(10, 5, 2))
  t <- torch_tensor(x)
  
  expect_equal(as.numeric(t), as.numeric(x), tolerance = 1e-7)
  expect_equal(as.integer(t), as.integer(x))
  expect_equal(as.double(t), as.double(x), tolerance = 1e-7)
  expect_equal(as.logical(t > 0.5), as.logical(x > 0.5))
})
