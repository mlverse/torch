context("optim-sgd")

test_that("optim_sgd works", {
  expect_optim_works(optim_sgd, list(lr = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1, nesterov = TRUE))
  expect_optim_works(optim_sgd, list(lr = 0.1, weight_decay = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1, dampening = 0.2))
})
