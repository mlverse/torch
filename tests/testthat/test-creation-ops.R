context("creation-ops")

test_that("torch_ones", {
  x <- torch_ones(5, 5,
    dtype = torch_float32(), layout = torch_strided(),
    device = torch_device("cpu"), requires_grad = TRUE
  )

  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))

  x <- torch_ones(c(5, 5))
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))

  x <- torch_ones(size = c(5, 5))
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))
})

test_that("ones_like", {
  x <- torch_ones(2, 2)
  y <- torch_ones_like(x)
  expect_equal_to_tensor(x, y)
})

test_that("rand", {
  x <- torch_rand(2, 2, 2)
  expect_equal(dim(as_array(x)), c(2, 2, 2))
})

test_that("rand_like", {
  x <- torch_rand(2, 2, 2)
  y <- torch_rand_like(x)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("randint", {
  x <- torch_randint(0, 10, c(2, 2))
  expect_equal(dim(as_array(x)), c(2, 2))

  x <- torch_randint(0, 10, c(2, 2), generator = torch_generator())
  expect_equal(dim(as_array(x)), c(2, 2))
})

test_that("randint_like", {
  x <- torch_randint(0, 10, c(2, 2))
  y <- torch_randint_like(x, 0, 500)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("randn", {
  x <- torch_randn(2, 2)
  expect_equal(dim(as_array(x)), c(2, 2))

  x <- torch_randn(c(2, 2), names = c("a", "b"))
  expect_equal(dim(as_array(x)), c(2, 2))
})

test_that("randn_like", {
  x <- torch_randn(2, 2)
  y <- torch_randn_like(x)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("randperm", {
  x <- torch_randperm(10)
  expect_equal(x$size(1), 10)
})

test_that("zeros", {
  x <- torch_zeros(2, 2)
  expect_equal(dim(as_array(x)), c(2, 2))

  x <- torch_zeros(c(2, 2), names = c("a", "b"))
  expect_equal(dim(as_array(x)), c(2, 2))
})

test_that("zeros_like", {
  x <- torch_zeros(2, 2, 2)
  y <- torch_zeros_like(x)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("empty", {
  x <- torch_zeros(2, 2)
  expect_equal(dim(as_array(x)), c(2, 2))

  x <- torch_zeros(c(2, 2), names = c("a", "b"))
  expect_equal(dim(as_array(x)), c(2, 2))
})

test_that("empty_like", {
  x <- torch_zeros(2, 2, 2)
  y <- torch_zeros_like(x)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("arange", {
  x <- torch_arange(1, 9)
  expect_equal(x$size(1), 9)

  x <- torch_arange(0, 2, 0.5)
  expect_equal(length(x), 5)

  x <- torch_arange(0, 2.5, 0.5)
  expect_equal(length(x), 6)

  x <- torch_arange(0, 2.5, 0.51)
  expect_equal(length(x), 5)

  x <- torch_arange(0, 2.5, 0.49)
  expect_equal(length(x), 6)

  x <- torch_arange(0, 1, 1)
  expect_equal(length(x), 2)

  x <- torch_arange(2.5, 1, -0.5)
  expect_equal(length(x), 4)

  x <- torch_arange(2.5, 1, -1)
  expect_equal(length(x), 2)

  x <- torch_arange(2.5, 1, -0.51)
  expect_equal(length(x), 3)

  x <- torch_arange(2.5, 1, -0.49)
  expect_equal(length(x), 4)

  x <- torch_arange(0, 1, 1, dtype = torch_float64())
  expect_equal(torch_finfo(x$dtype)$eps, torch_finfo(torch_double())$eps)

  x <- torch_arange(0, 1, 1, dtype = torch_float32())
  expect_equal(torch_finfo(x$dtype)$eps, torch_finfo(torch_float())$eps, tolerance = torch_finfo(torch_float())$eps)

  x <- torch_arange(0, 1, 1, dtype = torch_int64())
  expect_equal(torch_iinfo(x$dtype)$bits, 64)

  # deprecated
  expect_warning(x <- torch_range(1, 9))
  expect_equal(x$size(1), 9)
})

test_that("linspace", {
  x <- torch_linspace(1, 10, 100)
  expect_equal(x$size(1), 100)
})

test_that("logspace", {
  x <- torch_logspace(1, 10, 100)
  expect_equal(x$size(1), 100)
})

test_that("eye", {
  x <- torch_eye(10, 5)
  expect_equal_to_r(x, diag(nrow = 10, ncol = 5))

  x <- torch_eye(10)
  expect_equal_to_r(x, diag(nrow = 10, ncol = 10))
})

test_that("empty_strided", {
  x <- torch_empty_strided(c(2, 2), stride = c(1, 2))
  expect_equal(x$stride(1), 1)
  expect_equal(x$stride(2), 2)
})

test_that("full", {
  x <- torch_full(c(2, 2), fill_value = 2)
  expect_equal(dim(as_array(x)), c(2, 2))

  x <- torch_full(c(2, 2), fill_value = 2, names = c("a", "b"))
  expect_equal(dim(as_array(x)), c(2, 2))
})

test_that("full_like", {
  x <- torch_full(c(2, 2, 2), fill_value = 4)
  y <- torch_full_like(x, fill_value = 3)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("scalar_tensor", {
  x <- torch_scalar_tensor(1)
  expect_equal(length(x$shape), 0)
  expect_true(x$dtype == torch_float())

  x <- torch_scalar_tensor(1L)
  expect_equal(length(x$shape), 0)
  expect_true(x$dtype == torch_int64())

  x <- torch_tensor(1)
  x <- torch_scalar_tensor(x)
  expect_equal(length(x$shape), 0)
  expect_true(x$dtype == torch_float())
})
