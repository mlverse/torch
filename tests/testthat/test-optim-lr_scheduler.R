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

test_that("lr_reduce_on_plateau", {
  m <- nn_linear(10,10)
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, patience = 1, verbose=TRUE)
  
  val_loss <- 1
  expect_message(
  for (i in 1:5) {
    scheduler$step(val_loss)
  })
  expect_equal(o$param_groups[[1]]$lr, 1e-2) # matched to pytorch
  
  # test cooldown
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, patience = 1, cooldown=1) 
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 0.1) # matched to pytorch
  
  # test patience=0
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, patience = 0) 
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 1e-4) # matched to pytorch
  
  # test mode 'max'
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, mode='max', patience = 1)
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 1e-2) # matched to pytorch
  
  # test threshold mode abs
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, threshold_mode='abs', patience = 1)
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 1e-2) # matched to pytorch
  
  # test mode max and threshold mode abs
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.1, mode='max',
                                    threshold_mode='abs', patience = 1)
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 1e-2) # matched to pytorch
  
  # different factor
  o <- optim_adam(params = m$parameters, lr=1)
  scheduler <- lr_reduce_on_plateau(o, factor=0.3, patience = 1)
  for (i in 1:5) {
    scheduler$step(val_loss)
  }
  expect_equal(o$param_groups[[1]]$lr, 0.09) # matched to pytorch
})
