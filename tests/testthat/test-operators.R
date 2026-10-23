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

test_that("%*% works", {
  if (getRversion() < "4.3.0") {
    skip("%*% S3 operator for torch_tensor requires R >= 4.3.0")
  }

  lhs <- torch_tensor(matrix(c(1, 2, 3, 4), nrow = 2))
  rhs_tensor <- torch_tensor(matrix(c(0, 1, 1, 0), nrow = 2))
  rhs_matrix <- matrix(c(2, 0, 1, 2), nrow = 2)

  lhs_r <- as.matrix(as_array(lhs))
  rhs_tensor_r <- as.matrix(as_array(rhs_tensor))

  expect_equal_to_r(lhs %*% rhs_tensor, lhs_r %*% rhs_tensor_r)
  expect_equal_to_r(lhs %*% rhs_matrix, lhs_r %*% rhs_matrix)
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

test_that("%*% works on cuda", {
  if (getRversion() < "4.3.0") {
    skip("%*% S3 operator for torch_tensor requires R >= 4.3.0")
  }

  skip_if_cuda_not_available()

  lhs <- torch_tensor(matrix(c(1, 2, 3, 4), nrow = 2), device = "cuda")
  rhs_tensor <- torch_tensor(matrix(c(0, 1, 1, 0), nrow = 2), device = "cuda")
  rhs_matrix <- matrix(c(2, 0, 1, 2), nrow = 2)

  lhs_r <- as.matrix(as_array(lhs$cpu()))
  rhs_tensor_r <- as.matrix(as_array(rhs_tensor$cpu()))

  expect_equal_to_r(lhs %*% rhs_tensor, lhs_r %*% rhs_tensor_r)
  expect_equal_to_r(lhs %*% rhs_matrix, lhs_r %*% rhs_matrix)
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

test_that("lhs scalars promote like Python scalars", {
  x <- torch_ones(3, dtype = torch_bfloat16())
  for (r in list(1 + x, 1 - x, 2 * x, 1 / x, 2^x, 5 %% x, 5 %/% x)) {
    expect_true(r$dtype == torch_bfloat16())
    expect_equal(r$shape, 3L)
  }
  expect_equal_to_r((1 - x)$to(dtype = torch_float()), rep(0, 3))
  expect_equal_to_r((2 / x)$to(dtype = torch_float()), rep(2, 3))
  expect_equal_to_r((2^x)$to(dtype = torch_float()), rep(2, 3))
  expect_equal_to_r((5 %% x)$to(dtype = torch_float()), rep(0, 3))
  expect_equal_to_r((5 %/% x)$to(dtype = torch_float()), rep(5, 3))

  # an integer scalar keeps an integer tensor's dtype
  y <- torch_ones(3, dtype = torch_long())
  for (r in list(1L + y, 1L - y, 2L * y, 2L^y, 5L %% y, 5L %/% y)) {
    expect_true(r$dtype == torch_long())
  }
  expect_equal_to_r(5L %/% y, rep(5, 3))
  expect_equal_to_r(5L %% y, rep(0, 3))
  # while a double scalar promotes it to the default float dtype, as `y + 1` does
  expect_true((1 + y)$dtype == torch_float())
  expect_true((2.5 %/% y)$dtype == torch_float())
  expect_equal_to_r(2.5 %/% y, rep(2, 3))
  # division of integer tensors still promotes to the default dtype
  expect_true((1L / y)$dtype == torch_float())

  # dimensioned lhs operands keep dictating promotion and shape
  expect_true((c(1, 2, 3) - x)$dtype == torch_float())
  expect_tensor_shape(matrix(2) * torch_ones(3), c(1, 3))
  expect_true((matrix(2) * x)$dtype == torch_float())
})

test_that("lhs scalars behave like rhs scalars", {
  # a scalar and a 0-dim tensor give a 0-dim result on both sides
  z <- torch_ones(3, dtype = torch_bfloat16())$sum()
  expect_true((1 - z)$dtype == torch_bfloat16())
  expect_true((2 * z)$dtype == torch_bfloat16())
  expect_equal((1 - z)$shape, (z - 1)$shape)
  expect_equal((1 == z)$shape, (z == 1)$shape)
  expect_equal_to_r((1 - z)$to(dtype = torch_float()), -2)

  # a double scalar is no longer rounded through float32 first
  d <- torch_tensor(1, dtype = torch_float64())
  expect_equal(as.numeric(0.1 + d), as.numeric(d + 0.1), tolerance = 0)
  expect_equal(as.numeric(0.1 + d), 1.1, tolerance = 0)

  # an integer scalar outside the dtype's range is treated the same on both
  # sides. It currently wraps, as `u + 300L` already did; this records the
  # present behavior, not a decision to keep it.
  u <- torch_ones(3, dtype = torch_uint8())
  expect_true((300L + u)$dtype == torch_uint8())
  expect_equal(as.integer(as_array(300L + u)), as.integer(as_array(u + 300L)))
})

test_that("comparisons with a lhs scalar", {
  y <- torch_tensor(c(1, 2, 3), dtype = torch_long())
  expect_equal_to_r(2.5 > y, c(TRUE, TRUE, FALSE))
  expect_equal_to_r(2.5 < y, c(FALSE, FALSE, TRUE))
  expect_equal_to_r(2 >= y, c(TRUE, TRUE, FALSE))
  expect_equal_to_r(2 <= y, c(FALSE, TRUE, TRUE))
  expect_equal_to_r(2 == y, c(FALSE, TRUE, FALSE))
  expect_equal_to_r(2 != y, c(TRUE, FALSE, TRUE))
  expect_equal_to_r(y != c(1, 5, 3), c(FALSE, TRUE, FALSE))

  b <- torch_tensor(c(TRUE, FALSE))
  expect_equal_to_r(TRUE & b, c(TRUE, FALSE))
  expect_equal_to_r(FALSE | b, c(TRUE, FALSE))
})

test_that("lhs scalar division is computed at the promoted precision", {
  # integer inputs promote to the default float dtype before the scalar is used
  u <- torch_ones(1, dtype = torch_uint8())
  expect_true((300L / u)$dtype == torch_float())
  expect_equal_to_r(300L / u, 300)

  # a scalar too large for float16 still gives the representable quotient
  h <- torch_tensor(c(1024, 2048), dtype = torch_float16())
  r <- 65536 / h
  expect_true(r$dtype == torch_float16())
  expect_tensor_shape(r, 2)
  expect_equal_to_r(r$to(dtype = torch_float()), c(64, 32))

  # so does a denominator whose reciprocal overflows, in float16 and float32
  tiny <- torch_tensor(1e-5, dtype = torch_float16())
  r <- 1e-5 / tiny
  expect_true(r$dtype == torch_float16())
  expect_tensor_shape(r, 1)
  expect_true(as.logical(torch_isfinite(r)))
  expect_equal(as.numeric(r), 1, tolerance = 1e-2)
  expect_tensor_shape(1e-5 / tiny$squeeze(), integer(0))
  subnormal <- torch_tensor(1e-40)
  r <- 1e-10 / subnormal
  expect_true(r$dtype == torch_float())
  expect_true(as.logical(torch_isfinite(r)))
  expect_equal(as.numeric(r), 1e30, tolerance = 1e-4)

  # a double scalar outside float32's range is not narrowed before dividing
  r <- 1e40 / torch_tensor(1e10)
  expect_true(r$dtype == torch_float())
  expect_tensor_shape(r, 1)
  expect_equal(as.numeric(r), 1e30, tolerance = 1e-6)
  r <- 1e-50 / torch_tensor(1e-30)
  expect_true(r$dtype == torch_float())
  # expect_equal() compares values this small absolutely, so 0 would pass it
  expect_gt(as.numeric(r), 0)
  expect_lt(abs(as.numeric(r) / 1e-20 - 1), 1e-6)
  r <- 1e40 / torch_tensor(1e10, dtype = torch_bfloat16())
  expect_true(r$dtype == torch_bfloat16())
  expect_equal(as.numeric(r$to(dtype = torch_float())), 1e30, tolerance = 1e-2)
  x <- torch_tensor(1e10, requires_grad = TRUE)
  (1e40 / x)$backward()
  expect_true(x$grad$dtype == torch_float())
  expect_equal(as.numeric(x$grad), -1e20, tolerance = 1e-6)

  # float64 keeps the scalar's full precision
  d <- torch_tensor(3, dtype = torch_float64())
  expect_true((1 / d)$dtype == torch_float64())
  expect_equal(as.numeric(1 / d), 1 / 3, tolerance = 0)

  # gradients flow through the cast and keep the input dtype
  x <- torch_tensor(c(1024, 2048), dtype = torch_float16(), requires_grad = TRUE)
  (65536 / x)$sum()$backward()
  expect_true(x$grad$dtype == torch_float16())
  expect_equal_to_r(x$grad$to(dtype = torch_float()), -65536 / c(1024, 2048)^2)

  x <- torch_tensor(1e-5, dtype = torch_float16(), requires_grad = TRUE)
  y <- 1e-6 / x
  expect_true(as.logical(torch_isfinite(y)))
  expect_equal(as.numeric(y), 0.1, tolerance = 1e-2)
  y$backward()
  expect_true(as.logical(torch_isfinite(x$grad)))
  expect_equal(as.numeric(x$grad), -1e4, tolerance = 1e-2)

  x <- torch_tensor(c(2, 4), requires_grad = TRUE)
  (1 / x)$sum()$backward()
  expect_equal_to_r(x$grad, c(-0.25, -0.0625))
})

test_that("lhs scalars keep their truth value in logical operators", {
  u <- torch_ones(1, dtype = torch_uint8())
  expect_equal_to_r(256L & u, TRUE)
  expect_equal_to_r(u & 256L, TRUE)
  expect_equal_to_r(0 & u, FALSE)
  expect_equal_to_r(0 | u, TRUE)
  expect_equal_to_r(1e-8 | torch_zeros(1, dtype = torch_float16()), TRUE)
  expect_equal_to_r(torch_zeros(1, dtype = torch_float16()) | 1e-8, TRUE)
})
