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