test_that("optim_adam", {
  expect_optim_works(optim_adam, list(lr = 0.003))
  expect_optim_works(optim_adam, list(lr = 0.003, weight_decay = 0.1))
  expect_optim_works(optim_adam, list(lr = 0.003, weight_decay = 0.1, amsgrad = TRUE))
})
