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
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { print("hello")})
  y <- 2 * x
  expect_output(y$backward(), "hello")
  expect_equal_to_r(x$grad(), 2)
  
  # correctly sees the gradient
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { print(grad)})
  y <- 2 * x
  expect_output(y$backward(), "torch_tensor")
  
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { print("ABABA")})
  y <- 2 * x
  y$register_hook(function(grad) { print("EBEBE")})
  expect_output(y$backward(), "EBEBE.*ABABA")
})

test_that("register hook: can throw exceptions in the lantern thread", {
  skip_on_os("windows")
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { 2* grad})
  y <- 2 * x
  y$backward()
  expect_equal_to_r(x$grad(), 4)
  expect_error(y$backward())
})

test_that("register hook: can throw exceptions in the hook", {
  skip_on_os("windows")
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { stop()})
  y <- 2 * x
  expect_error(y$backward())
})

test_that("register_hook: grad non leaf", {
  # see https://github.com/pytorch/pytorch/blob/e0ee8000ac68ae58580ca62a59d5f40a9dd8710c/test/test_autograd.py#L400
  # This checks an edge case for register_hook.
  # We want to capture grad of a nonleaf tensor,
  # but avoid segfault during backward of other nonleaf tensors
  x <- torch_randn(5, options = list(requires_grad=TRUE))
  x_list <- x$unbind()
  x0 <- x_list[[1]]
  hook_results = NULL
  hook <- function(grad) {
    hook_results <<- grad
  }
  x0$register_hook(hook)
  x_list[[1]]$backward()
  
  expect_equal_to_r(hook_results, 1)
  expect_equal_to_r(x$grad(), c(1,0,0,0,0))
})



