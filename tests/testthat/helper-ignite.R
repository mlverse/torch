make_optimizer_maker = function(optimizer_fn) {
  function(..., steps = 2) {
    n <- nn_linear(1, 1)
    o <- optimizer_fn(n$parameters, ...)
    x <- torch_randn(10, 1)
    y <- torch_randn(10, 1)
    s <- function() {
      o$zero_grad()
      loss <- mean((n(x) - y)^2)
      loss$backward()
      o$step()
    }
    replicate(steps, s())
    o
  }
}

make_ignite_adamw <- make_optimizer_maker(optim_ignite_adamw)
make_ignite_sgd <- make_optimizer_maker(optim_ignite_sgd)
make_ignite_adam <- make_optimizer_maker(optim_ignite_adam)
make_ignite_rmsprop <- make_optimizer_maker(optim_ignite_rmsprop)
make_ignite_adagrad <- make_optimizer_maker(optim_ignite_adagrad)

sample_rmsprop_params <- function() {
  lr <- runif(1, 0.01, 0.02)
  alpha <- runif(1, 0.98, 0.99)
  eps <- runif(1, 0.0000001, 0.000002)
  weight_decay <- if (runif(1) < 0.5) runif(1, 0, 0.001) else 0
  momentum <- if (runif(1) < 0.5) runif(1, 0.8, 0.9) else 0
  centered <- sample(c(TRUE, FALSE), 1)
  list(lr = lr, alpha = alpha, eps = eps, weight_decay = weight_decay, momentum = momentum, centered = centered)
}

sample_adagrad_params <- function() {
  lr <- runif(1, 0.1, 0.2)
  weight_decay <- if (runif(1) < 0.5) runif(1, 0, 0.001) else 0
  lr_decay <- if (runif(1) < 0.5) runif(1, 0, 0.001) else 0
  initial_accumulator_value <- if (runif(1) < 0.5) runif(1, 0, 0.000001) else 0
  eps <- runif(1, 0.0000001, 0.000002)
  list(lr = lr, weight_decay = weight_decay, initial_accumulator_value = initial_accumulator_value, eps = eps)
}


sample_sgd_params <- function() {
  lr <- runif(1, 0.01, 0.02)
  nesterov <- sample(c(TRUE, FALSE), 1)
  dampening <- if (nesterov) 0 else runif(1, 0, 0.1)
  weight_decay <- if (runif(1) < 0.5) runif(1, 0, 0.001) else 0
  momentum <- if (nesterov || runif(1) < 0.5) runif(1, 0.8, 0.9) else 0
  list(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov)
}

sample_adam_params <- function() {
  lr <- runif(1, 0.01, 0.02)
  weight_decay <- if (runif(1) < 0.5) runif(1, 0, 0.001) else 0
  betas <- runif(2, 0.9, 0.99)
  eps <- runif(1, 0.001, 0.002)
  amsgrad <- sample(c(TRUE, FALSE), 1)
  list(lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = amsgrad)
}
sample_adamw_params <- sample_adam_params

expect_state_dict_works <- function(optimizer_fn, ...) {
  f <- function(load = FALSE) {
    n <- nn_linear(1, 1)
    o <- optimizer_fn(n$parameters, ...)
    x <- torch_randn(10, 1)
    y <- torch_randn(10, 1)
    n$parameters$bias$requires_grad_(FALSE)
    s <- function() {
      o$zero_grad()
      loss <- mean((n(x) - y)^2)
      loss$backward()
      o$step()
    }
    replicate(2, s())
    if (load) {
      o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
    }
    replicate(2, s())
    return(n$parameters)
  }
  w1 <- f(load = TRUE)
  w2 <- f(load = FALSE)
  expect_equal(w1, w2)
}

expect_ignite_can_change_param_groups <- function(optimizer_fn, ...) {
  n <- nn_linear(1, 1)
  o <- optimizer_fn(n$parameters, ...)
  for (nm in names(o$param_groups[-1L])) {
    if (is.numeric(o$param_groups[[nm]])) {
      o$param_groups[[nm]] = o$param_groups[[nm]] * 0.1
      expect_equal(o$param_groups[[nm]], o$param_groups[[nm]] * 0.1)
    } else if (is.logical(o$param_groups[[nm]])) {
      o$param_groups[[nm]] = !o$param_groups[[nm]]
      expect_equal(o$param_groups[[nm]], !o$param_groups[[nm]])
    } else {
      stop("Unknown type")
    }
  }
}

expect_ignite_can_add_param_group <- function(optimizer_fn, ...) {
  n <- nn_linear(1, 1)
  o <- optimizer_fn(n$parameters, lr = 0.1)
  n1 = nn_linear(1, 1)
  o$add_param_group(list(params = n1$parameters, lr = 19))
  expect_equal(o$param_groups[[1]]$lr, 0.1)
  expect_equal(o$param_groups[[2]]$lr, 19)
}
