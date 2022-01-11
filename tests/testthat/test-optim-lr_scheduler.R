test_that("lr_lambda", {
  m <- nn_linear(10, 10)
  o <- optim_sgd(params = m$parameters, lr = 1)
  scheduler <- lr_lambda(optimizer = o, lr_lambda = function(x) x)

  expect_equal(o$param_groups[[1]]$lr, 0)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 1)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 2)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 3)
})

test_that("lr_multiplicative", {
  m <- nn_linear(10, 10)
  o <- optim_sgd(params = m$parameters, lr = 1)
  scheduler <- lr_multiplicative(optimizer = o, lr_lambda = function(x) 0.95)

  expect_equal(o$param_groups[[1]]$lr, 1)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 0.95)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 0.95^2)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 0.95^3)
})

test_that("lr_one_cycle", {
  m <- nn_linear(10, 10)
  o <- optim_adam(params = m$parameters, lr = 1)

  expect_message({
    scheduler <- lr_one_cycle(
      optimizer = o, max_lr = 1, steps_per_epoch = 2, epochs = 3,
      verbose = TRUE, cycle_momentum = TRUE,
      div_factor = 1
    )

    for (i in 1:6) {
      scheduler$step()
    }
  })

  expect_equal(o$param_groups[[1]]$lr, 0.1335607, tol = 1e-6)
  expect_error(scheduler$step())
})

test_that("lr_step", {
  m <- nn_linear(10, 10)
  o <- optim_adam(params = m$parameters, lr = 1)
  scheduler <- lr_step(o, 1, gamma = 2)

  expect_equal(o$param_groups[[1]]$lr, 1)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 2)
  scheduler$step()
  expect_equal(o$param_groups[[1]]$lr, 4)
})
