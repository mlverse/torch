test_that("optim_asgd", {
  expect_optim_works(optim_asgd, list(lr = 0.1))
  expect_optim_works(optim_asgd, list(lr = 0.1, alpha = 0.7))
  expect_optim_works(optim_asgd, list(lr = 0.1, alpha = 0.7, t0 = 1e5))
  expect_optim_works(optim_asgd, list(lr = 0.1, weight_decay = 1e-5))
  expect_state_is_updated(optim_asgd)
})

test_that("ASGD updates the averaged parameter after averaging starts", {
  x <- torch_tensor(1, requires_grad = TRUE)
  opt <- optim_asgd(x, lr = 0.1, lambda = 0, t0 = 0)
  x$grad <- torch_tensor(1)
  for (i in seq_len(4)) opt$step()

  saved <- opt$state_dict()$state[["1"]]
  expect_equal(saved$step, 4)
  expect_equal(as.numeric(x), 0.6, tolerance = 1e-6)
  expect_equal(as.numeric(saved$ax), 0.7, tolerance = 1e-6)
  expect_equal(saved$mu, 0.25)
})
