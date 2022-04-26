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