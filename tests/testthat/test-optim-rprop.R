test_that("optim_rprop", {
  expect_optim_works(optim_rprop, list(lr = 0.1))
  expect_optim_works(optim_rprop, list(lr = 0.1, etas = c(0.6, 1.1)))
  expect_optim_works(optim_rprop, list(lr = 0.1, etas = c(0.6, 1.1), step_sizes = c(1e-6, 30)))
  expect_state_is_updated(optim_rprop)
})

test_that("Rprop preserves state when the gradient changes sign", {
  x <- torch_tensor(1, requires_grad = TRUE)
  opt <- optim_rprop(x, lr = 0.1)
  x$grad <- torch_tensor(1)
  opt$step()
  x$grad <- torch_tensor(-1)
  opt$step()

  saved <- opt$state_dict()$state[["1"]]
  expect_equal(saved$step, 2)
  expect_equal(as.numeric(x), 0.9, tolerance = 1e-6)
  expect_equal(as.numeric(saved$prev), 0)
  expect_equal(as.numeric(saved$step_size), 0.05, tolerance = 1e-6)
})
