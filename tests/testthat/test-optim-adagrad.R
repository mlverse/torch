test_that("optim_adagrad", {
  expect_optim_works(optim_adagrad, list(lr = 0.1))
  expect_optim_works(optim_adagrad, list(lr = 0.1, weight_decay = 1e-5))
  expect_optim_works(optim_adagrad, list(lr = 0.1, weight_decay = 1e-5, lr_decay = 1e-2))
  expect_optim_works(optim_adagrad, list(
    lr = 0.1, weight_decay = 1e-5, lr_decay = 1e-2,
    initial_accumulator_value = 1
  ))
  expect_state_is_updated(optim_adagrad)
})

test_that("Adagrad keeps independent accumulated state for each parameter", {
  x <- torch_tensor(1, requires_grad = TRUE)
  y <- torch_tensor(2, requires_grad = TRUE)
  opt <- optim_adagrad(list(x, y), lr = 0.1, lr_decay = 0.2,
                       initial_accumulator_value = 0.5)
  x$grad <- torch_tensor(1)
  opt$step()
  opt$step()
  y$grad <- torch_tensor(2)
  opt$step()

  saved <- opt$state_dict()
  expect_equal(saved$state[["1"]]$step, 3)
  expect_equal(saved$state[["2"]]$step, 1)
  expect_equal(as.numeric(saved$state[["1"]]$sum), 3.5)
  expect_equal(as.numeric(saved$state[["2"]]$sum), 4.5)
})
