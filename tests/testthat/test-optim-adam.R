test_that("optim_adam", {
  expect_optim_works(optim_adam, list(lr = 0.1))
  expect_optim_works(optim_adam, list(lr = 0.1, weight_decay = 1e-5))
  expect_optim_works(optim_adam, list(lr = 0.1, weight_decay = 1e-5, amsgrad = TRUE))
})
