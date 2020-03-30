test_that("can autograd", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  y <- 2 * x
  expect_invisible(y$backward())
  
  expect_equal_to_r(x$grad(), 2)
})

test_that("requires_grad works", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  expect_true(x$requires_grad())
  
  x <- torch_tensor(c(1), requires_grad = FALSE)
  expect_true(!x$requires_grad())
  
  x <- torch_tensor(c(1), requires_grad = FALSE)
  x$requires_grad_(TRUE)
  expect_true(x$requires_grad())
  x$requires_grad_(FALSE)
  expect_true(!x$requires_grad())
})

test_that("register_hook", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  x$register_hook(function(grad) { print("hello")})
  y <- 2 * x
  y$backward()
  x$grad()
})



