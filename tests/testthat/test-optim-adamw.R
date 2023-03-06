test_that("optim_adamw", {
  expect_optim_works(optim_adamw, list(lr = 0.1))
  expect_optim_works(optim_adamw, list(lr = 0.1, weight_decay = 1e-5))
  expect_optim_works(optim_adamw, list(lr = 0.1, weight_decay = 1e-5, amsgrad = TRUE))
  expect_state_is_updated(optim_adamw)
})
