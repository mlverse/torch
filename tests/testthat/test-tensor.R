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
  x <- c(1, 2, 3, 4)
  expect_equal_to_r(torch_tensor(x), x)

  x <- matrix(c(1, 2, 3, 4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)

  x <- array(c(1, 2, 3, 4, 5, 6, 7, 8), dim = c(2, 2, 2))
  expect_equal_to_r(torch_tensor(x), x)

  x <- c(NaN, -Inf, Inf)
  expect_equal_to_r(torch_tensor(x), x)
})

test_that("Integer tensors", {
  x <- 1:4
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)

  x <- matrix(c(1:4), ncol = 2)
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)

  x <- array(c(1:8), dim = c(2, 2, 2))
  expect_equal_to_r(torch_tensor(x)$to(dtype = torch_int()), x)

  x <- 1:5
  expect_equal_to_r(torch_tensor(1:5), x)

  x <- matrix(c(1:4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)

  x <- array(c(1:8), dim = c(2, 2, 2))
  expect_equal_to_r(torch_tensor(x), x)

  x <- 1:5
  expect_equal_to_r(torch_tensor(bit64::as.integer64(x)), x)

  x <- array(c(1:8), dim = c(2, 2, 2))
  o <- as.integer64(torch_tensor(x))
  expect_s3_class(o, "integer64")
  expect_s3_class(o, "array")
  expect_equal(dim(o), dim(x))

  x <- as.integer64(.Machine$integer.max) * 2
  y <- torch_tensor(x)
  z <- as.integer64(y)

  expect_equal(as.integer64(z), x)

  x <- torch_tensor(.Machine$integer.max)
  expect_equal(as.integer(x), .Machine$integer.max)
  expect_warning(as_array(2L * x))
  expect_warning(as.integer(2L * x))
})

test_that("Logical tensors", {
  x <- c(TRUE, TRUE, FALSE, FALSE)
  expect_equal_to_r(torch_tensor(x), x)

  x <- matrix(c(TRUE, TRUE, FALSE, FALSE), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)

  x <- array(c(rep(TRUE, 4), rep(FALSE, 4)), dim = c(2, 2, 2))
  expect_equal_to_r(torch_tensor(x), x)
})

test_that("Cuda tensor convertion", {
  skip_if_cuda_not_available()

  x <- torch_tensor(1, device = torch_device("cuda"))
  expect_error(as_array(x), regexp = NA)

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
  expect_true(x$is_cuda)
})

test_that("ndim", {
  x <- torch_randn(10, 10)
  expect_equal(x$ndim, 2)
})

test_that("as.matrix", {
  x <- torch_randn(2, 2)
  r <- as.matrix(x)
  expect_true("matrix" %in% class(r))
  expect_equal(dim(r), c(2, 2))

  x <- torch_randn(2, 2, 2)
  r <- as.matrix(x)
  expect_true("matrix" %in% class(r))
  expect_equal(dim(r), c(8, 1))
})

test_that("print tensor is truncated", {
  local_edition(3)
  expect_snapshot_output(torch_arange(0, 99))
  expect_snapshot_output(torch_arange(0, 24))
  expect_snapshot_output(print(torch_arange(0, 99), n = 50))
  expect_snapshot_output(print(torch_arange(0, 99), n = -1))
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
  x <- torch_randn(2, 2)
  expect_equal(x$has_names(), FALSE)
  expect_null(x$names)

  x <- torch_randn(2, 2, names = c("W", "H"))
  expect_equal(x$has_names(), TRUE)
  expect_equal(x$names, c("W", "H"))
})

test_that("rename works", {
  x <- torch_randn(2, 2, names = c("W", "H"))

  expect_equal(x$rename(W = "a")$names, c("a", "H"))
  expect_equal(x$rename(W = "a", H = "b")$names, c("a", "b"))
  expect_equal(x$rename("a", "b")$names, c("a", "b"))
  x$rename_(W = "a")
  expect_equal(x$names, c("a", "H"))

  x <- torch_randn(2, 2)
  expect_error(x$rename(W = "a"), class = "runtime_error")
  expect_equal(x$rename("a", "b")$names, c("a", "b"))
})

test_that("is_leaf", {
  a <- torch_rand(10, requires_grad = TRUE)
  expect_true(a$is_leaf)

  skip_if_cuda_not_available()
  a <- 2 * torch_rand(10, requires_grad = TRUE)$to(device = "cuda")
  expect_true(!a$is_leaf)
})

test_that("max and min", {
  x <- torch_tensor(1:10)

  expect_equal_to_r(x$min(), 1L)
  expect_equal_to_r(x$max(), 10L)

  expect_equal_to_r(x$min(other = 9L)[10], 9L)
  expect_equal_to_r(x$max(other = 2L)[1], 2L)

  x <- torch_tensor(
    rbind(
      c(1, 5, 0),
      c(2, 7, 9),
      c(5, 1, 4)
    )
  )
  expect_equal_to_r(x$min(dim = 2)[[1]], c(0, 2, 1))
  expect_equal_to_r(x$min(dim = 2)[[2]], c(3L, 1L, 2L))
  expect_tensor_shape(x$min(dim = 2, keepdim = TRUE)[[2]], c(3L, 1L))

  expect_equal_to_r(x$max(dim = 2)[[1]], c(5, 9, 5))
  expect_equal_to_r(x$max(dim = 2)[[2]], c(2L, 3L, 1L))
  expect_tensor_shape(x$min(dim = 2, keepdim = TRUE)[[2]], c(3L, 1L))

  expect_error(
    x$min(dim = 1, other = 2),
    class = "value_error"
  )


  x <- torch_tensor(c(1, 2))
  expect_equal_to_r(x$min(other = c(2, 1)), c(1, 1))
  expect_equal_to_r(x$max(other = c(2, 1)), c(2, 2))
})

test_that("element_size works", {
  x <- torch_tensor(1, dtype = torch_int8())
  result <- x$element_size()
  expect_equal(result, 1L)
  types <- list(
    torch_float32(),
    torch_float(),
    torch_float64(),
    torch_double(),
    torch_float16(),
    torch_half(),
    torch_uint8(),
    torch_int8(),
    torch_int16(),
    torch_short(),
    torch_int32(),
    torch_int(),
    torch_int64(),
    torch_long(),
    torch_bool(),
    torch_quint8(),
    torch_qint8(),
    torch_qint32()
  )
  for (type in types) {
    if (type == torch_quint8() || type == torch_qint8() || type == torch_qint32()) {
      x <- torch_tensor(1, dtype = torch_float())
      x <- torch_quantize_per_tensor(x, scale = 0.1, zero_point = 10, dtype = type)
    } else {
      x <- torch_tensor(1, dtype = type)
    }

    result <- x$element_size()
    expect_true(is.integer(result))
    expect_true(result > 0L)
    expect_true(length(result) == 1)
  }
})

test_that("tensor$bool works", {
  x <- torch_tensor(c(1, 0, 1))
  result <- x$bool()
  expected <- x$to(torch_bool())
  expect_equal_to_tensor(result, expected)

  expect_silent(
    result <- x$bool(memory_format = torch_contiguous_format())
  )
  expect_equal_to_tensor(result, expected)
})

test_that("copy_ respects the origin requires_grad", {
  x <- torch_randn(1, requires_grad = FALSE)
  y <- torch_randn(1, requires_grad = TRUE)

  x$copy_(y)
  expect_true(x$requires_grad)

  x <- torch_randn(1, requires_grad = FALSE)
  with_no_grad({
    x$copy_(y)
  })
  expect_false(x$requires_grad)

  x <- torch_randn(1, requires_grad = TRUE)
  y <- torch_randn(1, requires_grad = FALSE)
  expect_error(x$copy_(y))

  x <- torch_randn(1, requires_grad = TRUE)
  with_no_grad({
    x$copy_(y)
  })
  expect_true(x$requires_grad)

  x <- torch_randn(1, requires_grad = TRUE)
  y <- torch_randn(1, requires_grad = TRUE)
  with_no_grad({
    x$copy_(y)
  })
  expect_true(x$requires_grad)
})

test_that("size works", {
  x <- torch_randn(1, 2, 3, 4, 5)

  expect_equal(x$size(1), 1)
  expect_equal(x$size(-1), 5)
  expect_equal(x$size(-2), 4)
  expect_equal(x$size(4), 4)

  expect_error(x$size(0))
})

test_that("tensor identity works as expected", {
  skip_on_os("mac")

  v <- runif(1)
  gctorture()
  x <- torch_tensor(v)
  y <- x$abs_()$abs_()
  z <- x$abs_()
  gctorture(FALSE)

  expect_equal(rlang::obj_address(x), rlang::obj_address(y))
  expect_equal(rlang::obj_address(x), rlang::obj_address(z))

  rm(x)
  gc()

  class(y) <- class(torch_tensor(1))
  expect_equal_to_r(y, v, tolerance = 1e-7)

  x <- y$abs_()

  expect_equal(rlang::obj_address(x), rlang::obj_address(y))
})

test_that("print tensors with grad_fn and requires_grad", {
  x <- torch_tensor(c(1, 2, 3))
  y <- torch_tensor(c(1, 2, 3), requires_grad = TRUE)
  z <- torch_log(y)

  testthat::local_edition(3)
  expect_snapshot({
    print(x)
    print(y)
    print(z)
  })
})

test_that("using with optim", {
  skip_on_os("mac")

  expect_error(regexp = NA, {
    x <- torch_tensor(100, requires_grad = TRUE)
    opt <- optim_adam(x, lr = 1)
    l <- (2 * x^2)$mean()
    l$backward()
    gctorture(TRUE)
    opt$step()
    gctorture(FALSE)
  })
})

test_that("can create tensors from tensors", {
  x <- torch_tensor(1)
  y <- torch_tensor(x)
  expect_false(rlang::obj_address(x) == rlang::obj_address(y))
  expect_equal_to_tensor(x, y)
})

test_that("print complex tensors", {
  testthat::local_edition(3)
  skip_on_os("windows")
  skip_on_os("linux")
  x <- torch_complex(torch_randn(10), torch_randn(10))
  expect_snapshot(
    print(x)
  )

  x$requires_grad_(TRUE)
  expect_snapshot(
    print(x)
  )

  y <- 2*x
  expect_snapshot(
    print(y)
  )
})

test_that("complex tensors modifications and acessing", {

  real <- torch_randn(10, 10)
  imag <- torch_randn(10, 10)
  x <- torch_complex(real, imag)

  expect_equal_to_tensor(x$real, real)
  expect_equal_to_tensor(x$imag, imag)

  real <- torch_randn(10, 10)
  imag <- torch_randn(10, 10)
  x$real <- real
  x$imag <- imag

  expect_equal_to_tensor(x$real, real)
  expect_equal_to_tensor(x$imag, imag)

})

test_that("create complex from and to R", {

  x <- complex(real = c(0, 1), imaginary = c(1, 1))
  y <- torch_tensor(x)

  expect_equal_to_r(y$imag, Im(x))
  expect_equal_to_r(y$real, Re(x))
  expect_true(y$dtype == torch_cfloat())

  x <- complex(real = runif(1), imaginary = runif(1))
  y <- torch_tensor(x, dtype = torch_cdouble())

  expect_equal_to_r(y$imag, Im(x))
  expect_equal_to_r(y$real, Re(x))
  expect_true(y$dtype == torch_cdouble())

  x <- torch_complex(torch_randn(10, 10), torch_randn(10, 10))
  y <- as.array(x)
  z <- torch_tensor(y)

  expect_true(torch_allclose(x, z))

  x <- torch_complex(1, 1)
  y <- as.array(x)
  z <- torch_tensor(y)
  expect_true(torch_allclose(x, z))
  expect_equal(as.complex(x), complex(real = 1,imaginary = 1))

})

test_that("expand works", {
  x <- torch_tensor(1)
  y <- x$expand(c(1, 2, 3))
  expect_equal(y$size(), c(1, 2, 3))
  y <- x$expand(c(2, 2, 3), TRUE)
  expect_equal(y$size(), c(2, 2, 3))
  y <- x$expand(c(2, 2, 3), FALSE)
  expect_equal(y$size(), c(2, 2, 3))
})

test_that("narrow_copy works", {
  x <- torch_tensor(1:10)
  y <- torch_narrow_copy(x, dim = 1, start = 1, length = 3)
  expect_equal(y$size(), c(3))
  expect_equal(y, torch_tensor(2:4))
})

test_that("is_sparse works", {
  x <- torch_randn(5, 5)
  expect_false(x$is_sparse())
  x <- torch_sparse_coo_tensor(rbind(sample(10), sample(10)), rnorm(10))
  expect_true(x$is_sparse())
})

test_that("can make a byte tensor from a raw vector", {

  x <- charToRaw("hello world")
  ten <- torch_tensor(x)

  expect_equal(ten$dtype$.type(), "Byte")
  expect_equal(length(x), length(ten))

  expect_equal(as.array(ten), x)
  expect_equal(rawToChar(as.array(ten)), "hello world")
})

test_that("to can change both device and dtype", {

  x <- torch_randn(10, 10)
  y <- x$to(dtype = "double", device = "meta")

  expect_true(y$dtype == torch_double())
  expect_true(y$device == torch_device("meta"))
})

test_that("can convert to half using the method `half()`", {
  x <- torch_randn(10, 10)
  y <- x$half()

  expect_true(y$dtype == torch_half())

  x <- torch_tensor(1, dtype="half")
  expect_equal(as.numeric(x), 1)
})

test_that("can create tensor from a buffer", {
  x <- runif(10)
  y <- torch_tensor_from_buffer(x, shape = 10, dtype = "float64")
  expect_equal(as.numeric(y), x)
  y$add_(1)
  expect_equal(as.numeric(y), x)
})

test_that("can create a buffer from a tensor", {
  x <- torch_randn(10, 10)
  y <- buffer_from_torch_tensor(x)
  z <- torch_tensor_from_buffer(y, shape = c(10, 10), dtype="float")
  expect_true(torch_allclose(x, z))
})

test_that("can copy a mps tensor", {
  skip_if_not_m1_mac()
  x <- array(runif(100), dim = c(10, 10))
  y <- torch_tensor(x, device="mps")
  x_ <- as.array(y)

  expect_true(all.equal(x, x_, tolerance = 1e-5))
})

test_that("cloning works and preserves attributes", {
  # buffer
  x_buf <- nn_buffer(torch_tensor(1))
  x_buf_clone <- x_buf$clone()
  expect_equal(attributes(x_buf), attributes(x_buf_clone))
  x_buf[1] <- 2
  expect_false(torch_equal(x_buf, x_buf_clone))

  # parameter
  x_param <- nn_parameter(torch_tensor(1))
  x_param_clone <- x_param$clone()
  expect_equal(attributes(x_param), attributes(x_param_clone))
  x_param$requires_grad_(FALSE) # otherwise we cannot modify tensor in-place
  x_param[1] <- 2
  expect_false(torch_equal(x_param, x_param_clone))

  # tensor
  x <- torch_tensor(1)
  x_clone <- x$clone()
  expect_equal(attributes(x_clone), attributes(x))
  x[1] <- 2
  expect_false(torch_equal(x_clone, x))
})

test_that("requires grad is left unchanged when cloning tensor", {
  x_requires_grad = torch_tensor(1, requires_grad = TRUE)
  x_no_requires_grad <- torch_tensor(1, requires_grad = FALSE)
  expect_true(x_requires_grad$clone()$requires_grad)
  expect_false(x_no_requires_grad$clone()$requires_grad)
})

test_that("grad_fn and cloning", {
  # this is the same behaviour as shown by PyTorch's .clone() method
  x <- torch_tensor(1, requires_grad = TRUE)
  x1 <- x$clone()
  expect_true(grepl(pattern = "CloneBackward0", capture.output(x1$grad_fn), fixed = TRUE))
})

test_that("cuda tensor can be converted to tensor", {
  skip_if_cuda_not_available()
  x <- as.array(torch_tensor(1, device = "cuda"))
  expect_equal(x, 1)
})

test_that("detach preserves attributes (#1136)", {
  x <- nn_parameter(torch_tensor(1)$requires_grad_(TRUE))
  expect_true(inherits(x$detach(), "nn_parameter"))
})
