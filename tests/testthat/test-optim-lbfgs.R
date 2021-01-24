test_that("multiplication works", {
  expect_optim_works(optim_lbfgs, list(lr = 1))
  expect_optim_works(optim_lbfgs, list(lr = 1, max_iter = 50))
  expect_optim_works(optim_lbfgs, list(lr = 1, max_iter = 50, max_eval = 30))
  expect_optim_works(optim_lbfgs, list(lr = 1, max_iter = 50, history_size = 30))
})

test_that("state updates correctly", {
  
  x <- torch_tensor(1, requires_grad = TRUE)
  opt <- optim_lbfgs(x)
  
  opt$zero_grad()
  f <- function() {
    y <- 2 * x
  }
  
  opt$step(f)
  expect_equal(state(opt$param_groups[[1]]$params[[1]])$func_evals, 1)
  opt$step(f)
  expect_equal(state(opt$param_groups[[1]]$params[[1]])$func_evals, 2)
})

test_that("lbfgs works", {
  torch_manual_seed(1)
  x <- torch_randn(100, 10)
  y <- torch_sum(x, 2, keepdim = TRUE)
  
  model <- nn_linear(10, 1)
  optim <- optim_lbfgs(model$parameters, lr = 0.01)  
  
  fn <- function() {
    loss <- nnf_mse_loss(model(x), y)
    loss$backward()
    loss
  }
  
  for (i in 500) {
    optim$zero_grad()
    optim$step(fn)
  }
  
  expect_true(fn()$item() < 2)
  
})

test_that("lbfgs do not fail with NaN losses", {
  
  x <- torch_randn(100, 2)
  logits <- x[,1]*1 + x[,2]*1
  p <- torch_exp(logits)/(1 + torch_exp(logits))
  y <- (p > 0.5)$to(dtype = torch_long()) + 1L
  
  model <- nn_sequential(
    nn_linear(2,2),
    nn_softmax(dim = 2)
  )
  opt <- optim_lbfgs(model$parameters, lr = 0.1)
  
  clos <- function() {
    opt$zero_grad()
    loss <- nnf_nll_loss(torch_log(model(x)), y)
    loss$backward()
    loss
  }
  
  for (i in 1:10) {
    l <- opt$step(clos)
  }
  
  expect_equal_to_r(torch_isnan(l), TRUE)
  
})