test_that("stacks works", {
  
  stack <- Stack$new()
  
  x <- torch_tensor(c(1,2,3))
  y <- 1L
  z <- list(x, x)
  
  stack$push_back(x)
  stack$push_back(y)
  stack$push_back(z)
  
  x_ <- stack$at(1)
  y_ <- stack$at(2)
  z_ <- stack$at(3)
  
  expect_equal_to_tensor(x, x_)
  expect_equal(y, y_)
  expect_equal_to_tensor(z[[1]], z_[[1]])
  expect_equal_to_tensor(z[[2]], z_[[2]])
  expect_equal(length(z), length(z_))
  
})
