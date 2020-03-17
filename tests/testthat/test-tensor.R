test_that("Can create a tensor", {
  x <- torch_tensor(1)
  
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(1, dtype = torch_double())
  
  expect_s3_class(x, "torch_tensor")
})

test_that("Numeric tensors", {
  
  x <- c(1,2,3,4)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- matrix(c(1,2,3,4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(1,2,3,4,5,6,7,8), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
})

test_that("Integer tensors", {
  
  x <- 1:4
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- matrix(c(1:4), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(1:8), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
})

test_that("Logical tensors", {
  
  x <- c(TRUE, TRUE, FALSE, FALSE)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- matrix(c(TRUE, TRUE, FALSE, FALSE), ncol = 2)
  expect_equal_to_r(torch_tensor(x), x)
  
  x <- array(c(rep(TRUE, 4), rep(FALSE, 4)), dim = c(2,2,2))
  expect_equal_to_r(torch_tensor(x), x)
  
})
