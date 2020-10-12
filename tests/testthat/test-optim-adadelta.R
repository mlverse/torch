test_that("optim_adadelta", {
  expect_optim_works(optim_adadelta, list())
  expect_optim_works(optim_adadelta, list(rho = 0.9))
  expect_optim_works(optim_adadelta, list(rho = 0.9, weight_decay = 1e-5))
})
