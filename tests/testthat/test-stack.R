test_that("stacks works", {
  
  stack <- Stack$new()
  
  x <- torch_tensor(c(1,2,3))
  y <- 1L
  
  stack$push_back(x)
  stack$push_back(y)
  
  x_ <- stack$at(1)
  y_ <- stack$at(2)
  
  expect_equal_to_tensor(x, x_)
  expect_equal(y, y_)
  
})
