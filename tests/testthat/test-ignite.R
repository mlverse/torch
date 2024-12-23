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
  expect_equal(names(states), "1")
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

test_that("can initialize optimizer with different options per param group", {
  defaults = list(lr = 0.1, betas = c(0.9, 0.999), eps = 1e-8, weight_decay = 0, amsgrad = FALSE)
  # set args1 to slightly different values than defaults
  args1 = list(lr = 0.11, betas = c(0.91, 0.9991), eps = 1e-81, weight_decay = 0.1, amsgrad = TRUE)
  args2 = list(lr = 0.12, betas = c(0.92, 0.9992), eps = 1e-82, weight_decay = 0.2, amsgrad = FALSE)

  pgs = list(
    c(list(params = list(torch_tensor(1, requires_grad = TRUE))), args1),
    c(list(params = list(torch_tensor(2, requires_grad = TRUE))), args2),
    c(list(params = list(torch_tensor(3, requires_grad = TRUE))))
  )

  o = do.call(optim_ignite_adamw, args = c(list(params = pgs), defaults))
  expect_equal(o$state_dict()$state, set_names(list(), character()))
  step = function() {
    o$zero_grad()
    ((pgs[[1]]$params[[1]] * pgs[[2]]$params[[1]] * pgs[[3]]$params[[1]] * torch_tensor(1) - torch_tensor(2))^2)$backward()
    o$step()
  }
  replicate(3, step())
  pgs = o$param_groups
  expect_false(torch_equal(pgs[[1]]$params[[1]], torch_tensor(1)))
  expect_false(torch_equal(pgs[[2]]$params[[1]], torch_tensor(2)))
  expect_false(torch_equal(pgs[[3]]$params[[1]], torch_tensor(3)))
  sd = o$state_dict()
  expect_equal(sd$param_groups[[1]]$params, 1)
  expect_equal(sd$param_groups[[2]]$params, 2)
  expect_equal(sd$param_groups[[3]]$params, 3)
  pgs = o$param_groups
  pgs[[1]]$params = NULL
  pgs[[2]]$params = NULL
  pgs[[3]]$params = NULL
  expect_equal(pgs[[1]], args1[names(pgs[[1]])])
  expect_equal(pgs[[2]], args2[names(pgs[[2]])])
  expect_equal(pgs[[3]], defaults[names(pgs[[3]])])
})

test_that("params must have length > 1", {
  expect_error(optim_ignite_adamw(list()), "must have length")
})

test_that("can add a param group", {
  o = optim_ignite_adamw(list(torch_tensor(1)), lr = 5)
  o$add_param_group(list(params = list(torch_tensor(2)), lr = 10))
  expect_equal(o$param_groups[[1]]$params[[1]], torch_tensor(1))
  expect_equal(o$param_groups[[1]]$lr, 5)
  expect_equal(o$param_groups[[2]]$params[[1]], torch_tensor(2))
  expect_equal(o$param_groups[[2]]$lr, 10)
})

test_that("error handling when loading state dict", {
  o = make_ignite_adamw()
  expect_error(o$load_state_dict(list()), "must be a list with elements")
  sd1 = o$state_dict()
  sd1 = list(param_groups = sd1$param_groups, state = sd1$state[1])
  expect_error(o$load_state_dict(sd1), "To-be loaded state dict is missing states for parameters 2.", fixed = TRUE)
  sd2 = o$state_dict()
  sd2$state[[1]]$exp_avg = NULL
  expect_error(o$load_state_dict(sd2), "The 1-th state has elements with names exp_avg")
  sd3 = o$state_dict()
  sd3$param_groups[[1]]$lr = NULL
  expect_error(o$load_state_dict(sd3), "but got params, weight_decay")
})

test_that("can corectly load state dict for newly created optimizer", {
  old = make_ignite_adamw()
  new = make_ignite_adamw(steps = 0)
  new$load_state_dict(old$state_dict())
  expect_equal(old$state_dict(), new$state_dict())
  # tensor is copied
  expect_false(identical(old$state_dict()$state$`1`$exp_avg_sq, new$state_dict()$state$`1`$exp_avg_sq))
})
