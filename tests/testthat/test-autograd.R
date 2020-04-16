test_that("can autograd", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  y <- 2 * x
  
  expect_invisible(y$backward())
  expect_equal_to_r(x$grad(), 2)
})

test_that("can autograd with contexts", {

  with_no_grad({
    with_enable_grad({
      x <- torch_tensor(c(1), requires_grad = TRUE)
      y <- 2 * x
    })
  })

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
  skip("skip test")
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { 2* grad})
  y <- 2 * x
  y$backward()
  expect_equal_to_r(x$grad(), 4)
  expect_error(y$backward())
})

test_that("register hook: can throw exceptions in the hook", {
  skip("skip test")
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

test_that("register_hook: can call a the hook inside a hook", {
  
  v <- NULL
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  x$register_hook(function(grad) v <<- c(v, "x"))
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 2 * a
  a$register_hook(function(grad) {
    v <<- c(v, "a")
    y$backward()
  })
  a$backward()
  
  expect_equal(v, c("a", "x"))
  
  # add one more level of nesting  
  
  v <- NULL
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  x$register_hook(function(grad) v <<- c(v, "x"))
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 2 * a
  a$register_hook(function(grad) {
    v <<- c(v, "a")
    y$backward()
  })
  
  k <- torch_tensor(1, requires_grad = TRUE)
  l <- 2 * k
  k$register_hook(function(grad) {
    v <<- c(v, "k")
    a$backward()
  })
  
  l$backward()
  
  expect_equal(v, c("k", "a", "x"))
  
})

test_that("register_hook: can have 2 hooks that call backwards on different graphs", {
  
  v <- NULL
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  x$register_hook(function(grad) v <<- c(v, "x"))
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 2 * a
  a$register_hook(function(grad) {v <<- c(v, "a")})
  
  k <- torch_tensor(1, requires_grad = TRUE)
  l <- 2 * k
  k$register_hook(function(grad) {
    v <<- c(v, "k")
    a$backward()
  })
  l$register_hook(function(grad) {
    v <<- c(v, "l")
    y$backward()
  })
  
  l$backward()
  
  expect_equal(v, c("l", "x", "k", "a"))
})

test_that("remove_hook", {
  v <- NULL
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  
  hook <- x$register_hook(function(grad) v <<- c(v, "x"))
  y$backward()
  
  expect_equal(v, "x") # hook was added
  
  hook$remove()
  
  y <- 2 * x
  y$backward()
  
  expect_equal(v, "x") # hook has been removed
})

test_that("creating lambda functions", {
  
  f <- function(ctx, inputs) {
    ctx <- AutogradContext$new(ctx)
    x <- variable_list$new(ptr = inputs)$to_r()
    ctx$save_for_backward(x)
    out <- list(x[[1]] + 2* x[[2]] + x[[1]] * x[[2]])
    torch_variable_list(out)$ptr
  }
  
  b <- function(ctx, grad_output) {
    ctx <- AutogradContext$new(ctx)
    x <- variable_list$new(ptr = grad_output)$to_r()
    y <- ctx$get_saved_variables()
    print(y)
    torch_variable_list(list(
      x[[1]] + x[[1]]*y[[2]], x[[1]] + x[[1]] * y[[1]]
      ))$ptr
  }
  
  f_ <- cpp_Function_lambda(f)
  b_ <- cpp_Function_lambda(b)
  
  x <- torch_randn(c(5,5), options= list(requires_grad = TRUE))
  y <- torch_randn(c(5,5), options= list(requires_grad = TRUE))
  
  res <- cpp_Function_apply(torch_variable_list(list(x, y))$ptr, f_, b_)
  res <- variable_list$new(ptr = res)$to_r()
  go <- torch_ones(c(1), options = list(requires_grad = TRUE))
  s <- res[[1]]$sum()
  
  s$backward()
  
  y$grad()
  
})


test_that("custom autograd api", {
  
  forward <- function(ctx, var1, mul, var2) {
    ctx$save_for_backward(list(var1, var2))
    var1 + mul*var2 + var1 * var2
  }
  
  backward <- function(ctx, grad_output) {
    y <- ctx$get_saved_variables()
    x <- grad_output
    list(x[[1]] + x[[1]]*y[[2]], x[[1]] + x[[1]] * y[[1]])
  }
  
  custom <- autograd_function(forward, backward)
  
  x <- torch_randn(c(5,5), options= list(requires_grad = TRUE))
  y <- torch_randn(c(5,5), options= list(requires_grad = TRUE))
  
  res <- custom(var1 = x, mul = 2, var2 = y)
  go <- torch_ones(c(1), options = list(requires_grad = TRUE))
  s <- res[[1]]$sum()
  
  s$backward()
  
  y$grad()
  
})