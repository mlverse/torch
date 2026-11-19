test_that("optim_adadelta", {
  torch_manual_seed(1)
  expect_optim_works(optim_adadelta, list())
  expect_optim_works(optim_adadelta, list(rho = 0.9))
  expect_optim_works(optim_adadelta, list(rho = 0.9, weight_decay = 1e-5))
  expect_state_is_updated(optim_adadelta)
})

test_that("Adadelta keeps separate state for each parameter", {
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- torch_tensor(2, requires_grad = TRUE)
  opt <- optim_adadelta(list(x, y))
  x$grad <- torch_tensor(1)
  opt$step()
  opt$step()
  y$grad <- torch_tensor(1)
  opt$step()

  saved <- opt$state_dict()
  expect_equal(saved$state[["1"]]$step, 3)
  expect_equal(saved$state[["2"]]$step, 1)
  expect_true(is_torch_tensor(saved$state[["1"]]$square_avg))
  expect_true(is_torch_tensor(saved$state[["1"]]$acc_delta))
})
