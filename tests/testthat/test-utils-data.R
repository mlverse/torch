test_that("tensor dataset", {
  x <- torch_randn(1000, 10)
  y <- torch_randn(1000)
  
  data <- utils_dataset_tensor(x, y)

  expect_s3_class(data, "utils_dataset_tensor")
  expect_s3_class(data, "utils_dataset")
  expect_length(data, 1000)
  
  sub <- data[1:2]
  expect_tensor_shape(sub[[1]], c(2, 10))
  expect_tensor_shape(sub[[2]], c(2))
})
