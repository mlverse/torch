context("optim-sgd")

test_that("optim_sgd works", {
  expect_optim_works(optim_sgd, list(lr = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1, nesterov = TRUE))
  expect_optim_works(optim_sgd, list(lr = 0.1, weight_decay = 0.1))
  expect_optim_works(optim_sgd, list(lr = 0.1, momentum = 0.1, dampening = 0.2))
})

test_that("optim have classes", {
  
  expect_equal(
    class(optim_sgd),
    c("optim_sgd", "torch_optimizer_generator")
  )
  
  opt <- optim_sgd(lr = 0.1, list(torch_tensor(1, requires_grad = TRUE)))
  expect_equal(
    class(opt),
    c("optim_sgd", "torch_optimizer", "R6")
  )
  
  expect_true(is_optimizer(opt))
  
})

test_that("copy state between optimizers corecctly", {
  
  # start with a tensor and make one step in the optimize
  x <- torch_tensor(1, requires_grad = TRUE)
  
  opt <- optim_adam(x, lr = 0.1)
  (2*x)$backward()
  opt$step()
  opt$zero_grad()
  
  # now copy that tensor and its optimizer and make a step
  with_no_grad({
    y <- torch_empty(1, requires_grad = TRUE)$copy_(x)  
  })
  opt2 <- optim_adam(y, lr = 1) # use a different LR to make sure it is recovered
  opt2$load_state_dict(opt$state_dict())
  expect_equal(opt2$param_groups[[1]]$lr, 0.1)
  
  (2*y)$backward()
  opt2$step()
  opt2$state_dict()
  
  # another step in the original optimizer
  (2*x)$backward()
  opt$step()
  opt$zero_grad()
  
  expect_equal_to_tensor(x, y)
})