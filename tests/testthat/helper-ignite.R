expect_ignite_state_is_updated <- function(opt_fn) {
  x <- torch_tensor(1, requires_grad = TRUE)
  opt <- opt_fn(x)
  opt$zero_grad()
  y <- 2 * x
  y$backward()
  opt$step()
  expect_equal(as.numeric(opt$state_
  (opt$param_groups[[1]]$params[[1]])$step), 1)
  opt$step()
  expect_equal(as.numeric(opt$state$get(opt$param_groups[[1]]$params[[1]])$step), 2)

  state <- opt$state_dict()
  x2 <- torch_tensor(1, requires_grad = TRUE)
  opt2 <- opt_fn(x)
  opt2$load_state_dict(state)

  for (i in seq_along(opt$state$map)) {
    expect_equal(opt2$state$map[[i]], opt$state$map[[i]])
  }
}
