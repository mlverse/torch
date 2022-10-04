test_that("cudnn_available", {
  skip_if(!cuda_is_available())

  expect_true(backends_cudnn_is_available())
  expect_s3_class(backends_cudnn_version(), "numeric_version")
})

test_that("mps is available on M1 macs", {
  skip_if_not_m1_mac()
  expect_true(backends_mps_is_available())
})
