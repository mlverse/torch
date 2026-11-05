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
  expect_false(is.null(states[[1]]$exp_avg))
  expect_false(is.null(states[[1]]$exp_avg_sq))
  expect_false(is.null(states[[1]]$max_exp_avg_sq))
  expect_false(is.null(states[[1]]$step))
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

test_that("adam", {
  defaults <- sample_adam_params()
  expect_optim_works(optim_ignite_adam, defaults)
  expect_state_is_updated(optim_ignite_adam)
  o <- do.call(make_ignite_adam, defaults)
  if (length(o$state_dict()$state)) {
    expect_equal(names(o$state_dict()$state), c("1", "2"))
    expect_true(is_permutation(names(o$state_dict()$state[[1]]), c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")))
  }
  expect_equal(o$param_groups[[1]][-1L][names(defaults)], defaults)
  expect_ignite_can_change_param_groups(optim_ignite_adam)
  expect_ignite_can_add_param_group(optim_ignite_adam)
  do.call(expect_state_dict_works, c(list(optim_ignite_adam), defaults))
  # can save adam even when one of the tensors in the state is undefined in C++
  defaults$amsgrad <- FALSE
  o <- do.call(make_ignite_adam, defaults)
  prev <- o$state_dict()
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
  expect_equal(prev, o$state_dict())
})

test_that("adamw", {
  defaults <- sample_adamw_params()
  expect_optim_works(optim_ignite_adamw, defaults)
  expect_state_is_updated(optim_ignite_adamw)
  o <- do.call(make_ignite_adamw, defaults)
  if (length(o$state_dict()$state)) {
    expect_equal(names(o$state_dict()$state), c("1", "2"))
    expect_true(is_permutation(names(o$state_dict()$state[[1]]), c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")))
  }
  expect_equal(o$param_groups[[1]][-1L][names(defaults)], defaults)
  expect_ignite_can_change_param_groups(optim_ignite_adamw)
  expect_ignite_can_add_param_group(optim_ignite_adamw)
  do.call(expect_state_dict_works, c(list(optim_ignite_adamw), defaults))

  # can save adamw even when one of the tensors in the state is undefined in C++
  defaults$amsgrad <- FALSE
  o <- do.call(make_ignite_adamw, defaults)
  prev <- o$state_dict()
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
  expect_equal(prev, o$state_dict())
})

test_that("sgd", {
  defaults <- sample_sgd_params()
  expect_state_is_updated(optim_ignite_sgd, lr = 0.1, momentum = 0.9)
  o <- do.call(make_ignite_sgd, defaults)
  if (length(o$state_dict()$state)) {
    expect_equal(names(o$state_dict()$state), c("1", "2"))
    expect_true(is_permutation(names(o$state_dict()$state[[1]]), "momentum_buffer"))
  }
  expect_equal(o$param_groups[[1]][-1L][names(defaults)], defaults)
  expect_ignite_can_change_param_groups(optim_ignite_sgd, lr = 0.1)
  expect_ignite_can_add_param_group(optim_ignite_sgd)
  do.call(expect_state_dict_works, c(list(optim_ignite_sgd), defaults))
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))

  # saving of state dict
  o <- do.call(make_ignite_sgd, defaults)
  prev <- o$state_dict()
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
  expect_equal(prev, o$state_dict())
})

test_that("rmsprop", {
  defaults <- sample_rmsprop_params()
  expect_optim_works(optim_ignite_rmsprop, defaults)
  expect_state_is_updated(optim_ignite_rmsprop)
  o <- do.call(make_ignite_rmsprop, defaults)
  if (length(o$state_dict()$state)) {
    expect_equal(names(o$state_dict()$state), c("1", "2"))
    expect_true(is_permutation(names(o$state_dict()$state[[1]]), c("grad_avg", "square_avg", "momentum_buffer", "step")))
  }
  expect_equal(o$param_groups[[1]][-1L][names(defaults)], defaults)
  expect_ignite_can_change_param_groups(optim_ignite_rmsprop)
  expect_ignite_can_add_param_group(optim_ignite_rmsprop)
  do.call(expect_state_dict_works, c(list(optim_ignite_rmsprop), defaults))

  o <- do.call(make_ignite_rmsprop, defaults)
  prev <- o$state_dict()
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
  expect_equal(prev, o$state_dict())
})

test_that("adagrad", {
  defaults <- sample_adagrad_params()
  expect_optim_works(optim_ignite_adagrad, defaults)
  expect_state_is_updated(optim_ignite_adagrad)
  o <- do.call(make_ignite_adagrad, defaults)
  if (length(o$state_dict()$state)) {
    expect_equal(names(o$state_dict()$state), c("1", "2"))
    expect_true(is_permutation(names(o$state_dict()$state[[1]]), c("step", "sum")))
  }
  expect_equal(o$param_groups[[1]][-1L][names(defaults)], defaults)
  expect_ignite_can_change_param_groups(optim_ignite_adagrad)
  expect_ignite_can_add_param_group(optim_ignite_adagrad)
  do.call(expect_state_dict_works, c(list(optim_ignite_adagrad), defaults))

  o <- do.call(make_ignite_adagrad, defaults)
  prev <- o$state_dict()
  o$load_state_dict(torch_load(torch_serialize(o$state_dict())))
  expect_equal(prev, o$state_dict())
})

test_that("base class: can initialize optimizer with different options per param group", {
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

test_that("base class: params must have length > 1", {
  expect_error(optim_ignite_adamw(list()), "must have length")
})

test_that("base class: can change values of param_groups", {
  o = optim_ignite_adamw(list(torch_tensor(1, requires_grad = TRUE)), lr = 0.1)
  o$param_groups[[1]]$lr = 1
  expect_equal(o$param_groups[[1]]$lr, 1)
  o$param_groups[[1]]$amsgrad = FALSE
  expect_true(!o$param_groups[[1]]$amsgrad)
  o$param_groups[[1]]$amsgrad = TRUE
  expect_false(!o$param_groups[[1]]$amsgrad)
})


test_that("base class: error handling when loading state dict", {
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
  expect_error(o$load_state_dict(sd3), "must include names 'params")
})

test_that("base class: deep cloning not possible", {
  o = make_ignite_adamw(steps = 0)
  expect_error(o$clone(deep = TRUE), "OptimizerIgnite cannot be deep cloned")
})

test_that("base class: changing the learning rate has an effect", {
  n1 = nn_linear(1, 1)
  n2 = n1$clone(deep = TRUE)
  o1 = optim_sgd(n1$parameters, lr = 0.1)
  o2 = optim_sgd(n2$parameters, lr = 0.1)

  s = function(n, o) {
    o$zero_grad()
    ((n(torch_tensor(1)) - torch_tensor(1))^2)$backward()
    o$step()
  }

  s(n1, o1)
  s(n2, o2)
  expect_true(torch_equal(n1$parameters[[1]], n2$parameters[[1]]) && torch_equal(n1$parameters[[2]], n2$parameters[[2]]))
  o1$param_groups[[1]]$lr = 0.2
  s(n1, o1)
  s(n2, o2)
  expect_false(torch_equal(n1$parameters[[1]], n2$parameters[[1]]) && torch_equal(n1$parameters[[2]], n2$parameters[[2]]))
})


test_that("can specify additional param_groups", {
  o = optim_ignite_adamw(list(torch_tensor(1, requires_grad = TRUE)), lr = 0.1)
  o$param_groups[[1]]$initial_lr = 0.2
  expect_equal(o$param_groups[[1]]$initial_lr, 0.2)
  expect_equal(o$state_dict()$param_groups[[1]]$initial_lr, 0.2)
  o$param_groups[[1]]$initial_lr = 0.3
  expect_equal(o$param_groups[[1]]$initial_lr, 0.3)
  expect_equal(o$state_dict()$param_groups[[1]]$initial_lr, 0.3)

  o$param_groups[[1]]$initial_lr = NULL
  expect_equal(o$param_groups[[1]]$initial_lr, NULL)
  expect_equal(o$state_dict()$param_groups[[1]]$initial_lr, NULL)

  o = optim_ignite_adamw(params = list(
    list(params = list(torch_tensor(1, requires_grad = TRUE)), lr = 0.1),
    list(params = list(torch_tensor(1, requires_grad = TRUE)), lr = 0.2)
  ))

  o$param_groups[[1]]$initial_lr = 0.1
  o$param_groups[[2]]$initial_lr = 0.2
  expect_equal(o$param_groups[[1]]$initial_lr, 0.1)
  expect_equal(o$param_groups[[2]]$initial_lr, 0.2)
  expect_equal(o$state_dict()$param_groups[[1]]$initial_lr, 0.1)
  expect_equal(o$state_dict()$param_groups[[2]]$initial_lr, 0.2)
})
