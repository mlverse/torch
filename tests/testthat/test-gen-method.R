context("gen-method")

test_that("__and__", {
  x <- torch_tensor(TRUE)
  y <- x$`__and__`(x)
  expect_tensor(y)
  expect_equal_to_tensor(y, x)

  x <- torch_tensor(c(TRUE, FALSE))
  y <- x$`__and__`(TRUE)
  expect_tensor(y)
  expect_equal_to_tensor(y, x)
})

test_that("clone", {
  
  x <- torch_randn(10, 10)
  y <- x$clone()
  
  expect_equal_to_tensor(x, y)
  expect_true(! x$storage()$data_ptr() == y$storage()$data_ptr())
 
})

test_that("item", {
  
  x <- torch_tensor(1)
  expect_equal(x$item(), 1)
  
  x <- torch_tensor(1L)
  expect_equal(x$item(), 1L)
  
  x <- torch_tensor(TRUE)
  expect_equal(x$item(), TRUE)
  
  x <- torch_tensor(1.5)
  expect_equal(x$item(), 1.5)
  
  x <- torch_tensor(1.5, dtype = torch_double())
  expect_equal(x$item(), 1.5)
})
