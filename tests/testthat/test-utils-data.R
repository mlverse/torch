test_that("tensor dataset", {
  x <- torch_randn(1000, 10)
  y <- torch_randn(1000)
  
  data <- utils_dataset_tensor(x, y)

  expect_s3_class(data, "utils_dataset_tensor")
  expect_s3_class(data, "utils_dataset")
})
