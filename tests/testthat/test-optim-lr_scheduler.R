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



