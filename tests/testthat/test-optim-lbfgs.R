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
  expect_equal(opt$state$get(opt$param_groups[[1]]$params[[1]])$func_evals, 1)
  opt$step(f)
  expect_equal(opt$state$get(opt$param_groups[[1]]$params[[1]])$func_evals, 2)
})

test_that("lbfgs works", {
  torch_manual_seed(3)
  x <- torch_randn(100, 10)
  y <- torch_sum(x, 2, keepdim = TRUE)

  model <- nn_linear(10, 1)
  optim <- optim_lbfgs(model$parameters, lr = 0.01)

  fn <- function() {
    optim$zero_grad()
    loss <- nnf_mse_loss(model(x), y)
    loss$backward()
    loss
  }

  for (i in 500) {
    optim$step(fn)
  }

  expect_true(fn()$item() < 8)
})

test_that("lbfgs do not fail with NaN losses", {
  x <- torch_randn(100, 2)
  logits <- x[, 1] * 1 + x[, 2] * 1
  p <- torch_exp(logits) / (1 + torch_exp(logits))
  y <- (p > 0.5)$to(dtype = torch_long()) + 1L

  model <- nn_sequential(
    nn_linear(2, 2),
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

test_that("lbfgs works with strong wolfe", {
  torch_manual_seed(7)

  flower <- function(x) {
    torch_norm(x) + torch_sin(4 * torch_atan2(x[2], x[1]))
  }

  params <- torch_tensor(c(20, 20), requires_grad = TRUE)
  optimizer <- optim_lbfgs(params, line_search_fn = "strong_wolfe")

  calc_loss <- function() {
    optimizer$zero_grad()
    value <- flower(params)
    value$backward()
    value
  }

  optimizer$step(calc_loss)
  expect_lt(as.numeric(torch_linalg_norm(params, ord = Inf)), 0.1)
})

test_that("strong wolfe works", {
  torch_manual_seed(7)

  obj_func <- function(x, t, d) {
    new_x <- x + t * d
    list(new_x$square()$item(), 2 * new_x)
  }

  # exact solution
  ret <- .strong_wolfe(obj_func,
    x = torch_tensor(3),
    t = 1,
    d = torch_tensor(-1),
    f = 9,
    g = torch_tensor(6),
    gtd = torch_tensor(-6)
  )

  expect_equal(ret[[4]], 1)

  # wrong direction
  ret <- .strong_wolfe(obj_func,
    x = torch_tensor(3),
    t = 0.1,
    d = torch_tensor(1),
    f = 9,
    g = torch_tensor(6),
    gtd = torch_tensor(6)
  )

  expect_equal(ret[[3]], 0)

  # enters .cubic_interpolate in initial phase (tests for correct return type)
  ret <- .strong_wolfe(obj_func,
    x = torch_tensor(-3),
    t = 0.001,
    d = torch_tensor(1),
    f = 9,
    g = torch_tensor(-6),
    gtd = torch_tensor(-6)
  )

  expect_gt(ret[[4]], 1)
})
