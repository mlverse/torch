test_that("optim_rprop", {
  expect_optim_works(optim_rprop, list(lr = 0.1))
  expect_optim_works(optim_rprop, list(lr = 0.1, etas = c(0.6, 1.1)))
  expect_optim_works(optim_rprop, list(lr = 0.1, etas = c(0.6, 1.1), step_sizes = c(1e-6, 30)))
  expect_state_is_updated(optim_rprop)
})
