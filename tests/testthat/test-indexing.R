test_that("[ works", {
  x <- torch_randn(c(10,10,10))
  expect_equal(as_array(x[1,1,1]), as_array(x)[1,1,1])
  expect_equal(as_array(x[1,,]), as_array(x)[1,,])  
  expect_equal(as_array(x[1:5,,]), as_array(x)[1:5,,])
  expect_equal(as_array(x[1:10:2,,]), as_array(x)[seq(1,10, by = 2),,])
  
  x <- torch_tensor(0:9)
  expect_equal(as_array(x[-1]), 9)
  expect_equal(as_array(x[-2:10]), c(8,9))
  expect_equal(as_array(x[2:N]), c(1:9))
  
  x <- torch_randn(c(10,10,10,10))
  expect_equal(as_array(x[1,..]), as_array(x)[1,,,])
  expect_equal(as_array(x[1,1,..]), as_array(x)[1,1,,])
  expect_equal(as_array(x[..,1]), as_array(x)[,,,1])
  expect_equal(as_array(x[..,1,1]), as_array(x)[,,1,1])
  
  x <- torch_randn(c(10,10,10,10))
  i <- c(1,2,3,4)
  expect_equal(as_array(x[!!!i]), as_array(x)[1,2,3,4])
  i <- c(1,2)
  expect_equal(as_array(x[!!!i,3,4]), as_array(x)[1,2,3,4])
  
  x <- torch_tensor(1:10)
  y <- 1:10
  expect_equal_to_r(x[c(1,3,2,5)], y[c(1,3,2,5)])
  
  index <- 1:3
  expect_equal_to_r(x[index], y[index])
  
  x <- torch_randn(10, 10)
  x[c(2,3,1), c(3,2,1)]
  expect_length(x[c(2,3,1), c(3,2,1)], 3)
  
  x <- torch_randn(10)
  expect_equal_to_tensor(x[1:5,..], x[1:5])
  
  x <- torch_randn(10)
  expect_tensor_shape(x[, NULL], c(10, 1))
  expect_tensor_shape(x[NULL, , NULL], c(1, 10, 1))
  expect_tensor_shape(x[NULL, , NULL, NULL], c(1, 10, 1, 1))
  
  x <- torch_randn(10)
  expect_tensor_shape(x[, newaxis], c(10, 1))
  expect_tensor_shape(x[newaxis, , newaxis], c(1, 10, 1))
  expect_tensor_shape(x[newaxis, , newaxis, newaxis], c(1, 10, 1, 1))
  
  x <- torch_randn(10, 10)
  expect_tensor_shape(x[1,,drop=FALSE], c(1, 10))
  expect_tensor_shape(x[..,1,drop=FALSE], c(10, 1))
  expect_tensor_shape(x[..,-1,drop=FALSE], c(10, 1))
})

test_that("indexing error expectations", {
  skip_on_os("windows") # currently bombs on windows.
  
  x <- torch_randn(c(10,10,10,10))
  expect_error(x[1,], "incorrect number of dimensions", class = "value_error")
  expect_error(x[1,1,1,1,1], "too many indices", class = "c10::IndexError")
})
