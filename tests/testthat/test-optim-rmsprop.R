test_that("optim_rmsprop", {
  expect_optim_works(optim_rmsprop, list(lr = 0.1))
  expect_optim_works(optim_rmsprop, list(lr = 0.1, alpha = 0.8))
  expect_optim_works(optim_rmsprop, list(lr = 0.1, momentum = 0.1))
  expect_optim_works(optim_rmsprop, list(lr = 0.1, weight_decay = 0.1))
  expect_optim_works(optim_rmsprop, list(lr = 0.1, momentum = 0.1, centered = TRUE))
  expect_state_is_updated(optim_rmsprop)
})
