context("cuda")

test_that("cuda", {
  skip_if_cuda_not_available()

  expect_true(cuda_device_count() > 0)
  expect_true(cuda_current_device() >= 0)
  expect_true(cuda_is_available())

  capability <- cuda_get_device_capability(cuda_current_device())
  expect_type(capability, "integer")
  expect_length(capability, 2)
  expect_error(cuda_get_device_capability(cuda_device_count() + 1), "device must be an integer between 0 and")
})

test_that("cuda tensors", {
  skip_if_cuda_not_available()

  x <- torch_randn(10, 10, device = torch_device("cuda"))

  expect_equal(x$device$type, "cuda")
  expect_equal(x$device$index, 0)
})

test_that("cuda memory stats work", {
  skip_if_cuda_not_available()

  stats <- cuda_memory_stats()
  expect_length(stats, 13)
})
