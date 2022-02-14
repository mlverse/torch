context("device")

test_that("Can create devices", {
  device <- torch_device("cuda")
  expect_equal(device$type, "cuda")
  expect_null(device$index)

  device <- torch_device("cuda:1")
  expect_equal(device$type, "cuda")
  expect_equal(device$index, 1)

  device <- torch_device("cuda", 1)
  expect_equal(device$type, "cuda")
  expect_equal(device$index, 1)

  device <- torch_device("cpu", 0)
  expect_equal(device$type, "cpu")
  expect_equal(device$index, 0)

  skip_if_cuda_not_available()

  x <- torch_tensor(1, device = torch_device("cuda:0"))
  expect_equal(x$device$type, "cuda")
})

test_that("use string to define the device", {
  x <- torch_randn(10, 10, device = "cpu")
  expect_equal(x$device$type, "cpu")

  x <- torch_tensor(1, device = "cpu")
  expect_equal(x$device$type, "cpu")

  skip_if_cuda_not_available()

  x <- torch_tensor(1, device = "cuda")
  expect_equal(x$device$type, "cuda")
})

test_that("can compare devices", {
  x <- torch_randn(10, 10, device = "cpu")
  y <- torch_randn(10, 10, device = "cpu")
  z <- torch_randn(10, 10, device = "meta")

  expect_true(x$device == y$device)
  expect_false(x$device == z$device)
  expect_true(is_meta_device(z$device))
  expect_false(is_meta_device(x$device))

  skip_if_cuda_not_available()
  x <- torch_tensor(1, device = "cuda:0")
  y <- torch_tensor(1, device = "cpu")
  expect_false(x$device == y$device)
})

test_that("can print meta tensors", {
  x <- torch_randn(10, 10, device = "meta")
  expect_output(print(x), regexp = "META")
})
