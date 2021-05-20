context("utils-data")

test_that("tensor dataset", {
  x <- torch_randn(1000, 10)
  y <- torch_randn(1000)
  
  data <- tensor_dataset(x, y)

  expect_s3_class(data, "tensor_dataset")
  expect_s3_class(data, "dataset")
  expect_length(data, 1000)
  
  sub <- data[1:2]
  expect_tensor_shape(sub[[1]], c(2, 10))
  expect_tensor_shape(sub[[2]], c(2))
})

test_that("create a dataset with private and active methods", {
  
  ds <- dataset(
    "my_dataset",
    initialize = function() {
      
    },
    .getitem = function(i) {
      private$pvt_method()
    },
    .length = function() {
      self$actv_len
    },
    private = list(
      pvt_method = function() {
        1:10
      }
    ),
    active = list(
      actv_len = function() {
        10
      }
    )
  )
  
  d <- ds()
  expect_equal(d[1], 1:10)
  expect_equal(length(d), 10)
  
})

test_that("dataset_subset works", {
  x <- torch_randn(50, 10)
  y <- torch_randn(50)
  
  data <- tensor_dataset(x, y)
  data_subset <- dataset_subset(data, 1:14)
  expect_equal(length(data_subset), 14)
  expect_tensor(data_subset[1:4][[1]])
  expect_equal(nrow(data_subset[1:4][[1]]), 4)
})

test_that("getbatch will get a vector of integers", {
  
  data <- dataset(
    initialize = function() {},
    .getbatch = function(indexes) {
      expect_true(is.integer(indexes))
    },
    .length = function() {100}
  )
  
  expect_true(data()[list(1,2,3,4)])
})