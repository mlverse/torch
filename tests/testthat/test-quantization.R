context("quantization")

test_that("can create quantized tensors", {
  x <- torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
  expect_tensor(x)
})

test_that("is_quantized", {
  x <- torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
  expect_true(x$is_quantized())
  x <- torch_tensor(c(1, 2, 3))
  expect_true(!x$is_quantized())
})

test_that("dequantize quantized tensors", {
  x <- torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
  y <- x$dequantize()
  expect_true(!y$is_quantized())
  expect_true(x$is_quantized())

  y <- torch_dequantize(x)
  expect_true(!y$is_quantized())

  z <- torch_dequantize(list(x, x))
  expect_true(!z[[1]]$is_quantized())
  expect_true(!z[[2]]$is_quantized())
})

test_that("copy works", {
  a <- torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
  b <- torch_quantize_per_tensor(torch_tensor(c(1, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
  a$copy_(b)
  expect_equal_to_tensor(a, b)
})
