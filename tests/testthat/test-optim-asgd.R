test_that("optim_asgd", {
  expect_optim_works(optim_asgd, list(lr = 0.1))
  expect_optim_works(optim_asgd, list(lr = 0.1, alpha = 0.7))
  expect_optim_works(optim_asgd, list(lr = 0.1, alpha = 0.7, t0 = 1e5))
  expect_optim_works(optim_asgd, list(lr = 0.1, weight_decay = 1e-5))
  expect_state_is_updated(optim_asgd)
})
