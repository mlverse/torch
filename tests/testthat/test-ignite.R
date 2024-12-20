# Edge cases to test:
# - some parameters are only present for specific optimizer arguments (e.g. amsgrad)
#   also check that it works when amsgrad is TRUE for one param but FALSE for another
# - Some parameters are not trained at all
# In both cases above we need to ensure that the state is not corrupted
# most importantly, that the training can resume after loading the optimizer state.

test_that("un-optimized parameters and state dict", {
  w_true <- torch_randn(10, 1)
  x <- torch_randn(100, 10)
  y <- torch_mm(x, w_true)

  loss <- function(y, y_pred) {
    torch_mean(
      (y - y_pred)^2
    )
  }

  w <- torch_randn(10, 1, requires_grad = TRUE)
  z <- torch_randn(10, 1, requires_grad = TRUE)
  opt = optim_ignite_adamw(list(w, z), lr = 0.1)

  fn <- function() {
    opt$zero_grad()
    y_pred <- torch_mm(x, w)
    l <- loss(y, y_pred)
    l$backward()
    l
  }

  fn()
  opt$step()
  fn()
  opt$step()
  sd = opt$state_dict()
  expect_equal(names(sd), c("param_groups", "state"))
  states = sd$state
  # all parameters are included in the state dict even when they don't have a state.
  expect_false(cpp_tensor_is_undefined(states[[1]]$exp_avg))
  expect_false(cpp_tensor_is_undefined(states[[1]]$exp_avg_sq))
  expect_true(cpp_tensor_is_undefined(states[[1]]$max_exp_avg_sq))
  expect_false(cpp_tensor_is_undefined(states[[1]]$step))
  opt$load_state_dict(sd)
  x1 = unlist(states)
  x2 = unlist(opt$state_dict()$state)
  for (i in seq_along(x1)) {
    if (cpp_tensor_is_undefined(x1[[i]]) && cpp_tensor_is_undefined(x2[[i]])) {
      next
    }
    expect_equal(x1[[i]], x2[[i]])
  }
})

test_that("optim_adamw", {
  expect_optim_works(optim_ignite_adamw, list(lr = 0.1))
  expect_optim_works(optim_ignite_adamw, list(lr = 0.1, weight_decay = 0))
  expect_optim_works(optim_ignite_adamw, list(lr = 0.1, weight_decay = 1e-5, amsgrad = TRUE))
  expect_optim_works(optim_ignite_adamw, list(lr = 0.1, weight_decay = 1e-5, amsgrad = FALSE))
  expect_state_is_updated(optim_ignite_adamw)
})

make_adamw = function(...) {
  n = torch::nn_linear(1, 1)

  n$parameters[[1]]

  o = optim_ignite_adamw(n$parameters, ...)

  s = function() {
    x = torch_randn(10, 1)
    y = torch_randn(10, 1)
    loss = mean((n(x) - y)^2)
    loss$backward()
    o$step()
    o$zero_grad()
  }
  s()
  s()
  o$state_dict()
}

test_that("constructor arguments are passed to the optimizer", {
  n = nn_linear(1, 1)
  lr = 0.123
  weight_decay = 0.456
  betas = c(0.789, 0.444)
  eps = 0.0111

  o1 = optim_ignite_adamw(nn_linear(1, 1)$parameters,
    lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = TRUE)
  expect_equal(length(o1$param_groups), 1L)
  expect_equal(o1$param_groups[[1]][-1],
    list(lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = TRUE))

  o2 = optim_ignite_adamw(nn_linear(1, 1)$parameters,
    lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = FALSE)
  expect_equal(length(o2$param_groups), 1L)
  expect_equal(o2$param_groups[[1]][-1],
    list(lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = FALSE))
})

test_that("can change param groups", {
  o = optim_adamw(nn_linear(1, 1)$parameters)
  o$param_groups[[1]]$lr = 19
  expect_equal(o$param_groups[[1]]$lr, 19)
  o$param_groups[[1]]$betas = c(0.22, 0.33)
  expect_equal(o$param_groups[[1]]$betas, c(0.22, 0.33))
  o$param_groups[[1]]$eps = 0.5
  expect_equal(o$param_groups[[1]]$eps, 0.5)
  o$param_groups[[1]]$weight_decay = 0.77
  expect_equal(o$param_groups[[1]]$weight_decay, 0.77)
  o$param_groups[[1]]$amsgrad = FALSE
  expect_equal(o$param_groups[[1]]$amsgrad, FALSE)
  o$param_groups[[1]]$amsgrad = TRUE
  expect_equal(o$param_groups[[1]]$amsgrad, TRUE)
})
