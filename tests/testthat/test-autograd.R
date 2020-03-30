test_that("can autograd", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  y <- 2 * x
  y$backward()
  
  expect_equal_to_r(x$grad(), 2)
})


