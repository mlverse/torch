context("tensor")

test_that("Can create a tensor", {
  x <- torch_tensor(1)
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(1, dtype = torch_double())
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(numeric(), dtype = torch_float32())
  expect_equal(dim(x), 0)
  
  x <- torch_tensor(1)
  expect_true(x$dtype() == torch_float32())
  
  x <- torch_tensor(1, dtype = torch_double())
  expect_true(x$dtype() == torch_double())
  
  device <- x$device()
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
  expect_true(k$dtype() == torch_long())
})

test_that("cuda and cpu methods", {
  skip_if_cuda_not_available()
  
  x <- torch_tensor(1)
  y <- x$cuda()
  
  expect_true(y$device()$type == "cuda")
  
  # calling twice dont error
  y$cuda()
  expect_true(y$device()$type == "cuda")
  
  k <- y$cpu()
  
  expect_true(k$device()$type == "cpu")
  
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
