test_that("can autograd", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  y <- 2 * x
  
  expect_invisible(y$backward())
  expect_equal_to_r(x$grad, 2)
})

test_that("can autograd with contexts", {

  with_no_grad({
    with_enable_grad({
      x <- torch_tensor(c(1), requires_grad = TRUE)
      y <- 2 * x
    })
  })

  expect_invisible(y$backward())
  expect_equal_to_r(x$grad, 2)
})

test_that("requires_grad works", {
  x <- torch_tensor(c(1), requires_grad = TRUE)
  expect_true(x$requires_grad)
  
  x <- torch_tensor(c(1), requires_grad = FALSE)
  expect_true(!x$requires_grad)
  
  x <- torch_tensor(c(1), requires_grad = FALSE)
  x$requires_grad_(TRUE)
  expect_true(x$requires_grad)
  x$requires_grad_(FALSE)
  expect_true(!x$requires_grad)
})

test_that("register_hook", {
  x <- torch_tensor(c(2), requires_grad = TRUE)
  x$register_hook(function(grad) { print("hello")})
  y <- 2 * x
  expect_output(y$backward(), "hello")
  expect_equal_to_r(x$grad, 2)
  
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
  expect_equal_to_r(x$grad, 4)
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
  expect_equal_to_r(x$grad, c(1,0,0,0,0))
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


test_that("Simple autograd extension", {

  custom_pow <- autograd_function(
    forward = function(ctx, var1) {
      ctx$save_for_backward(var1)
      var1^2
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables[[1]]
      expect_tensor(v)
      list(var1 = 2*v)
    }
  )
  
  x <- torch_tensor(c(3), requires_grad = TRUE)
  out <- custom_pow(x)
  out$backward()
  
  expect_equal_to_r(out, 9)
  expect_equal_to_r(x$grad, 6)
  
  x <- torch_tensor(c(3), requires_grad = TRUE)
  out <- custom_pow(x)*custom_pow(x)
  out$backward()
  x$grad
  
  expect_equal_to_r(out, 81)
  expect_equal_to_r(x$grad, 12)
})

test_that("Autograd extension envolving 2 variables", {
  
  custom <- autograd_function(
    forward = function(ctx, a, b) {
      ctx$save_for_backward(a, b)
      a^2 + b^3
    },
    backward = function(ctx, grad_output) {
      a <- ctx$saved_variables[[1]]
      b <- ctx$saved_variables[[2]]
      expect_tensor(a)
      expect_tensor(b)
      # we must respect the named list and not the order.
      list(b = 3*(b^2), a = 2*a)
    }
  )
  x <- torch_tensor(c(3), requires_grad = TRUE)
  y <- torch_tensor(c(4), requires_grad = TRUE)
  out <- custom(x, y)
  out$backward()
  
  expect_equal_to_r(out, (3^2) + (4^3))
  expect_equal_to_r(x$grad, 2*3)
  expect_equal_to_r(y$grad, 3*(4^2))
  
  x <- torch_tensor(c(3), requires_grad = TRUE)
  y <- torch_tensor(c(4), requires_grad = TRUE)
  
  # order the order of the arguments must be respected even
  # even if they are passed in a different order in the
  # function call.
  out <- custom(b = y, a = x)
  out$backward()
  
  expect_equal_to_r(out, (3^2) + (4^3))
  expect_equal_to_r(x$grad, 2*3)
  expect_equal_to_r(y$grad, 3*(4^2))
})

test_that("Named values in saved variables", {
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1) {
      ctx$save_for_backward(var1 = var1, var2 = 2*var1)
      var1^2
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables
      expect_tensor(v$var1)
      expect_tensor(v$var2)
      
      list(var1 = v$var2)
    }
  )
  
  x <- torch_tensor(c(3), requires_grad = TRUE)
  out <- custom_pow(x)
  out$backward()
  
  expect_equal_to_r(out, 9)
  expect_equal_to_r(x$grad, 6)
})

test_that("Can have optional arguments in forward", {
  
  linear <- autograd_function(
    forward = function(ctx, input, weight, bias = NULL) {
      ctx$save_for_backward(input = input, weight = weight, bias = bias)
      output <- input$mm(weight$t())
      if (!is.null(bias))
        output <- output + bias$unsqueeze(0)$expand_as(output)
      
      output
    },
    backward = function(ctx, grad_output) {
      
      s <- ctx$saved_variables
      
      grads <- list(
        input = NULL,
        weight = NULL,
        bias = NULL
      )
      
      if (ctx$needs_input_grad$input)
        grads$input <- grad_output$mm(s$weight)
      
      if (ctx$needs_input_grad$weight)
        grads$weight <- grad_output$t()$mm(s$input)
      
      if (!is.null(s$bias) && ctx$needs_input_grad$bias)
        grads$bias <- grad_output$sum(dim = 0)
      
      grads
    }
  )
  
  x <- torch_tensor(matrix(c(1,1,1,1), ncol = 2))
  w <- torch_tensor(matrix(c(2,3), ncol = 2), requires_grad = TRUE)
  y <- torch_tensor(matrix(c(2,2), ncol = 1))
  
  o <- linear(x, w)
  l <- torch_mean((y - o)^2)
  l$backward()
  expect_equal_to_r(w$grad, matrix(c(6,6), ncol = 2))
  
  x <- torch_tensor(matrix(c(1,1,1,1), ncol = 2))
  w <- torch_tensor(matrix(c(2,3), ncol = 2), requires_grad = TRUE)
  b <- torch_tensor(0, requires_grad = TRUE)
  y <- torch_tensor(matrix(c(2,2), ncol = 1))
  
  o <- linear(x, w, b)
  l <- torch_mean((y - o)^2)
  l$backward()
  
  expect_equal_to_r(w$grad, matrix(c(6,6), ncol = 2))
  expect_equal_to_r(b$grad, 6)
})

test_that("Catch errors in forward and backward R functions", {
  skip_on_os("windows")
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1) {
      stop("stop forward")
      ctx$save_for_backward(var1)
      var1^2
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables[[1]]
      expect_tensor(v)
      list(var1 = 2*v)
    }
  )
  
  x <- torch_tensor(1, requires_grad = TRUE)
  expect_error(custom_pow(x), "stop forward")
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1) {
      ctx$save_for_backward(var1)
      var1^2
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables[[1]]
      expect_tensor(v)
      stop("stop backward")
      list(var1 = 2*v)
    }
  )
  
  r <- custom_pow(x)
  expect_error(r$backward(), "stop backward")
})

test_that("Can pass constants to save_for_backward", {
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1, i) {
      ctx$save_for_backward(var1 = var1, i = i)
      var1^i
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables
      expect_tensor(v$var1)
      expect_is(v$i, "numeric")
      expect_named(v, c("var1", "i"))
      list(var1 = v$i*(v$var1^(v$i - 1)))
    }
  )
  
  x <- torch_tensor(2, requires_grad = TRUE)
  r <- custom_pow(x, 2)
  r$backward()
  
  expect_equal_to_r(r, 4)
  expect_equal_to_r(x$grad, 2*2)
  
  x <- torch_tensor(2, requires_grad = TRUE)
  r <- custom_pow(x, 3)
  r$backward()
  
  expect_equal_to_r(r, 8)
  expect_equal_to_r(x$grad, 3*2^(3-1))
  
})

test_that("Forward can return a list", {
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1, i) {
      ctx$save_for_backward(var1 = var1, i = i)
      list(var1^i, var1^(i+1))
    },
    backward = function(ctx, grad_output) {
      v <- ctx$saved_variables
      expect_tensor(grad_output[[1]])
      expect_tensor(grad_output[[2]])
      list(var1 = v$i*(v$var1^(v$i - 1)))
    }
  )
  
  x <- torch_tensor(2, requires_grad = TRUE)
  r <- custom_pow(x, 2)
  r[[1]]$backward()
  expect_equal_to_r(x$grad, 4)
  
  custom_pow <- autograd_function(
    forward = function(ctx, var1, i) {
      ctx$save_for_backward(var1 = var1, i = i)
      list(var1^i, var1^(i+1))
    },
    backward = function(ctx, out1, out2) {
      v <- ctx$saved_variables
      expect_tensor(out1)
      expect_tensor(out2)
      list(var1 = v$i*(v$var1^(v$i - 1)))
    }
  )
  
  x <- torch_tensor(2, requires_grad = TRUE)
  r <- custom_pow(x, 2)
  r[[1]]$backward()
  expect_equal_to_r(x$grad, 4)
})

test_that("can use mark_dirty", {
  
  # https://github.com/pytorch/pytorch/blob/master/test/test_autograd.py#L1936
  
  inplace <- autograd_function(
    forward = function(ctx, a, b) {
      ctx$mark_dirty(a)
      list(a$add_(b), b + 2)
    },
    backward = function(ctx, ga, gb) {
      list(a = ga, b = ga + gb)
    }
  )
  
  x <- torch_tensor(2)
  y <- torch_tensor(3, requires_grad = TRUE)
  r <- inplace(x, y)
  expect_equal_to_tensor(r[[1]], x)
  expect_true(r[[1]]$requires_grad)
  r[[1]]$backward()
  expect_equal_to_r(y$grad, 1)
  
  # https://github.com/pytorch/pytorch/blob/master/test/test_autograd.py#L1388
  double_in_place <- autograd_function(
    forward = function(ctx, x) {
      x$mul_(2)
      ctx$mark_dirty(x)
      list(x, x)
    },
    backward = function(ctx, g1, g2) {
      list(x = g1 * 2 + g2 * 2)
    }
  )
  
  x <- torch_tensor(5, requires_grad = TRUE)
  skip_on_os("windows")
  expect_error(double_in_place(x), "leaf Variable that requires grad")
})

test_that("mark_non_differentiable", {
 
  # https://github.com/pytorch/pytorch/blob/master/test/test_autograd.py#L1309
  
  myfun <- autograd_function(
    forward = function(ctx, input) {
      output <- input > 0
      ctx$mark_non_differentiable(output)
      output
    },
    backward = function(ctx, g) {
      list(input = g * 0)
    }
  )
  
  x <- torch_tensor(c(-1, 2), requires_grad = TRUE)
  mask <- myfun(x)
  
  expect_false(mask$requires_grad)
  y <- x$masked_fill(mask, 0)
  expect_no_error(y$sum()$backward())
  
  myfun <- autograd_function(
    forward = function(ctx, input) {
      a <- input + 1
      b <- input + 2
      ctx$mark_non_differentiable(a)
      list(a, b)
    },
    backward = function(ctx, ga, gb) {
      expect_equal_to_r(ga, 0)
      expect_equal_to_r(gb, 1)
      list(input = gb)
    }
  )
  
  x <- torch_tensor(1, requires_grad = TRUE)
  r <- myfun(x)
  expect_false(r[[1]]$requires_grad)
  expect_true(r[[2]]$requires_grad)
  r[[2]]$backward()
  expect_equal_to_r(x$grad, 1)
})

test_that("retain_grad is invisible", {
  x <- torch_tensor(1, requires_grad = TRUE)
  expect_invisible(x$retain_grad())
})

test_that("grad_fn works", {
  x <- torch_ones(c(2,2), options = list(requires_grad = TRUE))
  y <- x$mean()
  expect_output(print(y$grad_fn), "MeanBackward0")
  k <- y$grad_fn$next_functions
  expect_length(k, 1)
  l <- k$next_functions
  expect_length(l, 0)
})

test_that("autograd_backward", {
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 3 * a
  
  on <- torch_ones(c(1))
  autograd_backward(list(y, b), list(on, on))
  
  expect_equal_to_r(x$grad, 2)
  expect_equal_to_r(a$grad, 3)
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 3 * a
  
  on <- torch_ones(c(1))
  autograd_backward(list(y, b))
  
  expect_equal_to_r(x$grad, 2)
  expect_equal_to_r(a$grad, 3)
  
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  
  a <- torch_tensor(1, requires_grad = TRUE)
  b <- 3 * a
  
  on <- torch_ones(c(1))
  autograd_backward(list(y, b), list(NULL, on))
  
  expect_equal_to_r(x$grad, 2)
  expect_equal_to_r(a$grad, 3)
})

test_that("autograd_backward works for single tensors", {
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- 2 * x
  
  autograd_backward(y)
  expect_equal_to_r(x$grad, 2)
  
  x <- torch_tensor(1, requires_grad = TRUE)
  on <- torch_tensor(1)
  y <- 2 * x
  
  autograd_backward(y, on)
  expect_equal_to_r(x$grad, 2)
})
