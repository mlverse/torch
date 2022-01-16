expect_optim_works <- function(optim, defaults) {
  w_true <- torch_randn(10, 1)
  x <- torch_randn(100, 10)
  y <- torch_mm(x, w_true)

  loss <- function(y, y_pred) {
    torch_mean(
      (y - y_pred)^2
    )
  }

  w <- torch_randn(10, 1, requires_grad = TRUE)
  z <- torch_randn(10, 1, requires_grad = TRUE)
  defaults[["params"]] <- list(w, z)
  opt <- do.call(optim, defaults)

  fn <- function() {
    opt$zero_grad()
    y_pred <- torch_mm(x, w)
    l <- loss(y, y_pred)
    l$backward()
    l
  }

  initial_value <- fn()

  for (i in seq_len(200)) {
    opt$step(fn)
  }


  expect_true(as_array(fn()) <= as_array(initial_value) / 2)

  opt$state_dict()
}

expect_state_is_updated <- function(opt_fn) {
  x <- torch_tensor(1, requires_grad = TRUE)
  opt <- opt_fn(x)
  opt$zero_grad()
  y <- 2 * x
  y$backward()
  opt$step()
  expect_equal(opt$state$get(opt$param_groups[[1]]$params[[1]])$step, 1)
  opt$step()
  expect_equal(opt$state$get(opt$param_groups[[1]]$params[[1]])$step, 2)

  state <- opt$state_dict()
  x2 <- torch_tensor(1, requires_grad = TRUE)
  opt2 <- opt_fn(x)
  opt2$load_state_dict(state)

  for (i in seq_along(opt$state$map)) {
    expect_equal(opt2$state$map[[i]], opt$state$map[[i]])
  }
}
