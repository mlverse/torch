context("cuda")

test_that("cuda", {
  skip_if_cuda_not_available()
 
  expect_true(cuda_device_count() > 0)
  expect_true(cuda_current_device() >= 0)
  expect_true(cuda_is_available())
})

test_that("cuda tensors", {
  skip_if_cuda_not_available()
  
  x <- torch_randn(10, 10, device = torch_device("cuda"))
  
  expect_equal(x$device()$type, "cuda")
  expect_equal(x$device()$index, 0)
})
