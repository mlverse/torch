test_that("optim_adadelta", {
  expect_optim_works(optim_adadelta, list(lr = 0.1))
  expect_optim_works(optim_adadelta, list(lr = 0.1, rho = 0.7, weight_decay = 1e-5))
  expect_optim_works(optim_adadelta, list(lr = 0.1, rho = 0.7, weight_decay = 1e-5))
})
