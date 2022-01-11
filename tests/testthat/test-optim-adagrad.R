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
