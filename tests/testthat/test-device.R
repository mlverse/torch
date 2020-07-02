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
  expect_equal(x$device()$type, "cuda")
})

test_that("use string to define the device", {
  
  x <- torch_randn(10, 10, device = "cpu")
  expect_equal(x$device()$type, "cpu")
  
  x <- torch_tensor(1, device = "cpu")
  expect_equal(x$device()$type, "cpu")
  
  skip_if_cuda_not_available()
  
  x <- torch_tensor(1, device = "cuda")
  expect_equal(x$device()$type, "cuda")
})
