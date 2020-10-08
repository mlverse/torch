context("tensor")

test_that("Can create a tensor", {
  x <- torch_tensor(1)
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(1, dtype = torch_double())
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(numeric(), dtype = torch_float32())
  expect_equal(dim(x), 0)
  
  x <- torch_tensor(1)
  expect_true(x$dtype == torch_float32())
  
  x <- torch_tensor(1, dtype = torch_double())
  expect_true(x$dtype == torch_double())
  
  device <- x$device
  expect_equal(device$type, "cpu")
})

test_that("Numeric tensors", {
  
  x <- c(1,2,3,4)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- matrix(c(1,2,3,4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(1,2,3,4,5,6,7,8), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- c(NaN, -Inf, Inf)
  expect_equal_to_r(torch_tensor(x), x)
  
})

test_that("Integer tensors", {
  
  x <- 1:4
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)
  
  x <- matrix(c(1:4), ncol = 2)
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)
  
  x <- array(c(1:8), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)
  
  x <- 1:5
  expect_equal_to_r(torch_tensor(1:5), x)
  
  x <- matrix(c(1:4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(1:8), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- 1:5
  expect_equal_to_r(torch_tensor(bit64::as.integer64(x)), x)
  
  x <- array(c(1:8), dim = c(2,2,2))
  o <- as.integer64(torch_tensor(x))
  expect_s3_class(o, "integer64")
  expect_s3_class(o, "array")
  expect_equal(dim(o), dim(x))
  
  x <- as.integer64(.Machine$integer)*2
  y <- torch_tensor(x)
  z <- as.integer64(y)
  
  expect_equal(as.integer64(z), x)
  
})

test_that("Logical tensors", {
  
  x <- c(TRUE, TRUE, FALSE, FALSE)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- matrix(c(TRUE, TRUE, FALSE, FALSE), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(rep(TRUE, 4), rep(FALSE, 4)), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
})

test_that("Cuda tensor convertion", {
  skip_if_cuda_not_available()
  
  x <- torch_tensor(1, device = torch_device("cuda"))
  expect_error(as_array(x), class = "runtime_error")
  
  x <- x$to(dtype = torch_float(), device = torch_device("cpu"))
  expect_equal_to_r(x, 1)
})

test_that("Pass only device argument to `to`", {
  x <- torch_tensor(1)
  expect_tensor(x$to(dtype = torch_int()))
  expect_tensor(x$to(device = torch_device("cpu")))
  expect_tensor(x$to(device = torch_device("cpu"), dtype = torch_int()))
  
  y <- torch_tensor(1, dtype = torch_long())
  k <- x$to(other = y)
  expect_true(k$dtype == torch_long())
})

test_that("cuda and cpu methods", {
  skip_if_cuda_not_available()
  
  x <- torch_tensor(1)
  y <- x$cuda()
  
  expect_true(y$device$type == "cuda")
  
  # calling twice dont error
  y$cuda()
  expect_true(y$device$type == "cuda")
  
  k <- y$cpu()
  
  expect_true(k$device$type == "cpu")
  
})

test_that("stride", {
  x <- torch_randn(10, 10)
  expect_identical(x$stride(), c(10, 1))
  expect_identical(x$stride(1), 10)
  expect_identical(x$stride(2), 1)
})

test_that("is_contiguous", {
  
  x <- torch_randn(10, 10)
  expect_true(x$is_contiguous())
  x$t_()
  expect_true(!x$is_contiguous())
  
})

test_that("is_cuda", {
  x <- torch_randn(10, 10)
  expect_true(!x$is_cuda)
  
  skip_if_cuda_not_available()
  
  x <- torch_randn(10, 10, device = torch_device("cuda"))
  expect_true(X$is_cuda)
})

test_that("ndim", {
  
  x <- torch_randn(10, 10)
  expect_equal(x$ndim, 2)
  
})

test_that("as.matrix", {
  
  x <- torch_randn(2,2)
  r <- as.matrix(x)
  expect_equal(class(r), c("matrix", "array"))
  expect_equal(dim(r), c(2,2))
  
  x <- torch_randn(2,2,2)
  r <- as.matrix(x)
  expect_equal(class(r), c("matrix", "array"))
  expect_equal(dim(r), c(8,1))
  
})

test_that("print tensor is truncated", {
  
  expect_known_value(torch_arange(0, 100), file = "assets/print1")
  expect_known_value(torch_arange(0, 25), file = "assets/print2")
  expect_known_value(print(torch_arange(0, 100), n = 50), file = "assets/print3")
  expect_known_value(print(torch_arange(0, 100), n = -1), file = "assets/print4")
  
})

test_that("scatter works", {
  
  index <- torch_tensor(matrix(c(1, 2), ncol = 1), dtype = torch_long())
  
  z <- torch_zeros(3, 5)
  expected_out <- torch_zeros(3, 5)
  expected_out[1:2, 1] <- 1L
  
  expect_equal_to_tensor(z$scatter(1, index, 1), expected_out)
  expect_equal_to_tensor(z$scatter_(1, index, 1), expected_out)
  
})

test_that("names and has_names", {
  
  x <- torch_randn(2,2)
  expect_equal(x$has_names(), FALSE)
  expect_null(x$names)
  
  x <- torch_randn(2,2, names = c("W", "H"))
  expect_equal(x$has_names(), TRUE)
  expect_equal(x$names, c("W", "H"))
  
})
