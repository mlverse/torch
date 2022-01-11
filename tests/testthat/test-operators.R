context("operators")

test_that("+ works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x + x, 2 * c(1, 2, 3, 4))
  expect_equal_to_r(x + 2, c(1, 2, 3, 4) + 2)
  expect_equal_to_r(2 + x, c(1, 2, 3, 4) + 2)
})

test_that("- works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x - x, rep(0, 4))
  expect_equal_to_r(x - 2, c(1, 2, 3, 4) - 2)
  expect_equal_to_r(2 - x, 2 - c(1, 2, 3, 4))
  expect_equal_to_r(-x, -c(1, 2, 3, 4))
})

test_that("* works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x * x, c(1, 2, 3, 4)^2)
  expect_equal_to_r(x * 2, c(1, 2, 3, 4) * 2)
  expect_equal_to_r(2 * x, c(1, 2, 3, 4) * 2)
})

test_that("/ works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x / x, rep(1, 4))
  expect_equal_to_r(x / 2, c(1, 2, 3, 4) / 2)
  expect_equal_to_r(12 / x, 12 / c(1, 2, 3, 4))
})

test_that("^ works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x^x, c(1, 2, 3, 4)^c(1, 2, 3, 4))
  expect_equal_to_r(x^2, c(1, 2, 3, 4)^2)
  expect_equal_to_r(2^x, 2^c(1, 2, 3, 4))
})

test_that("%% works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x %% x, c(1, 2, 3, 4) %% c(1, 2, 3, 4))
  expect_equal_to_r(x %% 2, c(1, 2, 3, 4) %% 2)
  expect_equal_to_r(2 %% x, 2 %% c(1, 2, 3, 4))
})

test_that("%/% works", {
  x <- torch_tensor(c(1, 2, 3, 4))
  expect_equal_to_r(x %/% x, c(1, 2, 3, 4) %/% c(1, 2, 3, 4))
  expect_equal_to_r(x %/% 2, c(1, 2, 3, 4) %/% 2)
  expect_equal_to_r(2 %/% x, 2 %/% c(1, 2, 3, 4))
})

test_that("> works", {
  x <- torch_tensor(c(2, 2))
  y <- torch_tensor(c(2, 3))
  expect_equal_to_r(x > y, c(FALSE, FALSE))
  expect_equal_to_r(y > x, c(FALSE, TRUE))
  expect_equal_to_r(y > 2, c(FALSE, TRUE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that(">= works", {
  x <- torch_tensor(c(2, 2))
  y <- torch_tensor(c(2, 3))
  expect_equal_to_r(x >= y, c(TRUE, FALSE))
  expect_equal_to_r(y >= x, c(TRUE, TRUE))
  expect_equal_to_r(y >= 2, c(TRUE, TRUE))
  expect_equal_to_r(2 >= y, c(TRUE, FALSE))
})

test_that("< works", {
  x <- torch_tensor(c(2, 2))
  y <- torch_tensor(c(2, 3))
  expect_equal_to_r(x < y, c(FALSE, TRUE))
  expect_equal_to_r(y < x, c(FALSE, FALSE))
  expect_equal_to_r(y < 2, c(FALSE, FALSE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that("<= works", {
  x <- torch_tensor(c(2, 2))
  y <- torch_tensor(c(2, 3))
  expect_equal_to_r(x <= y, c(TRUE, TRUE))
  expect_equal_to_r(y <= x, c(TRUE, FALSE))
  expect_equal_to_r(y <= 2, c(TRUE, FALSE))
  expect_equal_to_r(2 <= y, c(TRUE, TRUE))
})

test_that("== works", {
  x <- torch_tensor(c(1, 2))
  y <- torch_tensor(c(2, 2))
  expect_equal_to_r(x == y, c(FALSE, TRUE))
  expect_equal_to_r(x == 2, c(FALSE, TRUE))
})

test_that("!= works", {
  x <- torch_tensor(c(1, 2))
  y <- torch_tensor(c(2, 2))
  expect_equal_to_r(x != y, c(TRUE, FALSE))
  expect_equal_to_r(x != 2, c(TRUE, FALSE))
})

test_that("& works", {
  x <- torch_tensor(c(1, 2))
  y <- torch_tensor(c(2, 2))
  expect_equal_to_r(x & y, c(TRUE, TRUE))
  expect_equal_to_r(x & 0, c(FALSE, FALSE))
})

test_that("| works", {
  x <- torch_tensor(c(0, 2))
  y <- torch_tensor(c(1, 2))
  expect_equal_to_r(x | y, c(TRUE, TRUE))
  expect_equal_to_r(x | 0, c(FALSE, TRUE))
})

test_that("! works", {
  x <- torch_tensor(c(1, 2))
  y <- torch_tensor(c(0, 2))
  expect_equal_to_r(!x, c(FALSE, FALSE))
  expect_equal_to_r(!y, c(TRUE, FALSE))
})

test_that("dim works", {
  x <- torch_randn(c(2, 2))
  expect_equal(dim(x), c(2, 2))
})

test_that("length works", {
  x <- torch_randn(c(2, 2))
  expect_equal(length(x), 4)
})

test_that("as.*", {
  x <- array(runif(200), dim = c(10, 5, 2))
  t <- torch_tensor(x)

  expect_equal(as.numeric(t), as.numeric(x), tolerance = 1e-5)
  expect_equal(as.integer(t), as.integer(x))
  expect_equal(as.double(t), as.double(x), tolerance = 1e-5)
  expect_equal(as.logical(t > 0.5), as.logical(x > 0.5))
})

test_that("abs works", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(abs(torch_tensor(x))), as.numeric(abs(x)), tolerance = 1e-5)
})

test_that("sign works", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(sign(torch_tensor(x))), as.numeric(sign(x)), tolerance = 1e-5)
})

test_that("sqrt works", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(sqrt(torch_tensor(x))), as.numeric(sqrt(x)), tolerance = 1e-5)
})

test_that("ceiling, floor, trunc work", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(ceiling(torch_tensor(x))), as.numeric(ceiling(x)), tolerance = 1e-5)
  expect_equal(as.numeric(floor(torch_tensor(x))), as.numeric(floor(x)), tolerance = 1e-5)
  expect_equal(as.numeric(trunc(torch_tensor(x))), as.numeric(trunc(x)), tolerance = 1e-5)
})

test_that("cumsum works", {
  x <- runif(200)
  expect_equal(as.numeric(cumsum(torch_tensor(x))), as.numeric(cumsum(x)), tolerance = 1e-5)
})

test_that("log* work", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(log(torch_tensor(x))), as.numeric(log(x)), tolerance = 1e-5)
  expect_equal(as.numeric(log(torch_tensor(x), base = 3)), as.numeric(log(x, base = 3)), tolerance = 1e-5)
  expect_equal(as.numeric(log2(torch_tensor(x))), as.numeric(log2(x)), tolerance = 1e-5)
  expect_equal(as.numeric(log10(torch_tensor(x))), as.numeric(log10(x)), tolerance = 1e-5)
  expect_equal(as.numeric(log1p(torch_tensor(x))), as.numeric(log1p(x)), tolerance = 1e-5)
})

test_that("acos, asin, atan, cos*, sin*, tan* work", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(acos(torch_tensor(x))), as.numeric(acos(x)), tolerance = 1e-5)
  expect_equal(as.numeric(asin(torch_tensor(x))), as.numeric(asin(x)), tolerance = 1e-5)
  expect_equal(as.numeric(atan(torch_tensor(x))), as.numeric(atan(x)), tolerance = 1e-5)
  expect_equal(as.numeric(cos(torch_tensor(x))), as.numeric(cos(x)), tolerance = 1e-5)
  expect_equal(as.numeric(cosh(torch_tensor(x))), as.numeric(cosh(x)), tolerance = 1e-5)
  expect_equal(as.numeric(sin(torch_tensor(x))), as.numeric(sin(x)), tolerance = 1e-5)
  expect_equal(as.numeric(sinh(torch_tensor(x))), as.numeric(sinh(x)), tolerance = 1e-5)
  expect_equal(as.numeric(tan(torch_tensor(x))), as.numeric(tan(x)), tolerance = 1e-5)
  expect_equal(as.numeric(tanh(torch_tensor(x))), as.numeric(tanh(x)), tolerance = 1e-5)
})

test_that("exp* work", {
  x <- array(runif(200), dim = c(10, 5, 2))
  expect_equal(as.numeric(exp(torch_tensor(x))), as.numeric(exp(x)), tolerance = 1e-5)
  expect_equal(as.numeric(expm1(torch_tensor(x))), as.numeric(expm1(x)), tolerance = 1e-5)
})

test_that("max min work", {
  x <- array(runif(200), dim = c(10, 5, 2))
  y <- c(1, 2, 3)
  expect_equal(as.numeric(max(torch_tensor(x))), max(x), tolerance = 1e-5)
  expect_equal(as.numeric(max(torch_tensor(x), torch_tensor(y))), max(x, y), tolerance = 1e-5)
  expect_equal(as.numeric(min(torch_tensor(x))), min(x), tolerance = 1e-5)
  expect_equal(as.numeric(min(torch_tensor(x), torch_tensor(y))), min(x, y), tolerance = 1e-5)
})

test_that("prod works", {
  x <- array(runif(200), dim = c(10, 5, 2))
  y <- c(1, 2, 3)
  expect_equal(as.numeric(prod(torch_tensor(x))), prod(x))
  expect_equal(as.numeric(prod(torch_tensor(x), dim = 1)), as.numeric(torch_prod(x, dim = 1)), tolerance = 1e-5)
  expect_equal(as.numeric(prod(torch_tensor(x), torch_tensor(y))), prod(x, y), tolerance = 1e-5)
})

test_that("sum works", {
  x <- array(runif(200), dim = c(10, 5, 2))
  y <- c(1, 2, 3)
  expect_equal(as.numeric(sum(torch_tensor(x))), sum(x), tolerance = 1e-5)
  expect_equal(as.numeric(sum(torch_tensor(x), dim = 1)), as.numeric(torch_sum(x, dim = 1)), tolerance = 1e-5)
  expect_equal(as.numeric(sum(torch_tensor(x), torch_tensor(y))), sum(x, y), tolerance = 1e-5)
})

# GPU tests ---------------------------------------------------------------

test_that("+ works", {
  skip_if_cuda_not_available()

  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x + x, 2 * c(1, 2, 3, 4))
  expect_equal_to_r(x + 2, c(1, 2, 3, 4) + 2)
  expect_equal_to_r(2 + x, c(1, 2, 3, 4) + 2)
})

test_that("- works", {
  skip_if_cuda_not_available()

  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x - x, rep(0, 4))
  expect_equal_to_r(x - 2, c(1, 2, 3, 4) - 2)
  expect_equal_to_r(2 - x, 2 - c(1, 2, 3, 4))
  expect_equal_to_r(-x, -c(1, 2, 3, 4))
})

test_that("* works", {
  skip_if_cuda_not_available()

  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x * x, c(1, 2, 3, 4)^2)
  expect_equal_to_r(x * 2, c(1, 2, 3, 4) * 2)
  expect_equal_to_r(2 * x, c(1, 2, 3, 4) * 2)
})

test_that("/ works", {
  skip_if_cuda_not_available()

  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x / x, rep(1, 4))
  expect_equal_to_r(x / 2, c(1, 2, 3, 4) / 2)
  expect_equal_to_r(12 / x, 12 / c(1, 2, 3, 4))
})

test_that("^ works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x^x, c(1, 2, 3, 4)^c(1, 2, 3, 4))
  expect_equal_to_r(x^2, c(1, 2, 3, 4)^2)
  expect_equal_to_r(2^x, 2^c(1, 2, 3, 4))
})

test_that("%% works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x %% x, c(1, 2, 3, 4) %% c(1, 2, 3, 4))
  expect_equal_to_r(x %% 2, c(1, 2, 3, 4) %% 2)
  expect_equal_to_r(2 %% x, 2 %% c(1, 2, 3, 4))
})

test_that("%/% works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2, 3, 4), device = "cuda")
  expect_equal_to_r(x %/% x, c(1, 2, 3, 4) %/% c(1, 2, 3, 4))
  expect_equal_to_r(x %/% 2, c(1, 2, 3, 4) %/% 2)
  expect_equal_to_r(2 %/% x, 2 %/% c(1, 2, 3, 4))
})

test_that("> works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(2, 2), device = "cuda")
  y <- torch_tensor(c(2, 3), device = "cuda")
  expect_equal_to_r(x > y, c(FALSE, FALSE))
  expect_equal_to_r(y > x, c(FALSE, TRUE))
  expect_equal_to_r(y > 2, c(FALSE, TRUE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that(">= works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(2, 2), device = "cuda")
  y <- torch_tensor(c(2, 3), device = "cuda")
  expect_equal_to_r(x >= y, c(TRUE, FALSE))
  expect_equal_to_r(y >= x, c(TRUE, TRUE))
  expect_equal_to_r(y >= 2, c(TRUE, TRUE))
  expect_equal_to_r(2 >= y, c(TRUE, FALSE))
})

test_that("< works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(2, 2), device = "cuda")
  y <- torch_tensor(c(2, 3), device = "cuda")
  expect_equal_to_r(x < y, c(FALSE, TRUE))
  expect_equal_to_r(y < x, c(FALSE, FALSE))
  expect_equal_to_r(y < 2, c(FALSE, FALSE))
  expect_equal_to_r(2 > y, c(FALSE, FALSE))
})

test_that("<= works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(2, 2), device = "cuda")
  y <- torch_tensor(c(2, 3), device = "cuda")
  expect_equal_to_r(x <= y, c(TRUE, TRUE))
  expect_equal_to_r(y <= x, c(TRUE, FALSE))
  expect_equal_to_r(y <= 2, c(TRUE, FALSE))
  expect_equal_to_r(2 <= y, c(TRUE, TRUE))
})

test_that("== works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2), device = "cuda")
  y <- torch_tensor(c(2, 2), device = "cuda")
  expect_equal_to_r(x == y, c(FALSE, TRUE))
  expect_equal_to_r(x == 2, c(FALSE, TRUE))
})

test_that("!= works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2), device = "cuda")
  y <- torch_tensor(c(2, 2), device = "cuda")
  expect_equal_to_r(x != y, c(TRUE, FALSE))
  expect_equal_to_r(x != 2, c(TRUE, FALSE))
})

test_that("& works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(1, 2), device = "cuda")
  y <- torch_tensor(c(2, 2), device = "cuda")
  expect_equal_to_r(x & y, c(TRUE, TRUE))
  expect_equal_to_r(x & 0, c(FALSE, FALSE))
})

test_that("| works", {
  skip_if_cuda_not_available()
  x <- torch_tensor(c(0, 2), device = "cuda")
  y <- torch_tensor(c(1, 2), device = "cuda")
  expect_equal_to_r(x | y, c(TRUE, TRUE))
  expect_equal_to_r(c(TRUE, TRUE) | y, c(TRUE, TRUE))
  expect_equal_to_r(x | 0, c(FALSE, TRUE))
})

test_that("mean works", {
  x <- c(1, 2, 3, 4)

  expect_equal_to_r(
    mean(torch_tensor(x)),
    2.5
  )

  x <- torch_randn(20, 100)

  expect_tensor_shape(mean(x, dim = 1), 100)
  expect_tensor_shape(mean(x, dim = 2), 20)
  expect_tensor_shape(mean(x, dim = 1, keepdim = TRUE), c(1, 100))
  expect_tensor_shape(mean(x, dim = 2, keepdim = TRUE), c(20, 1))
})
