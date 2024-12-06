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
  o
}

test_that("params are part of the param_groups", {

})

test_that("constructor arguments are passed to the optimizer", {
  n = nn_linear(1, 1)
  lr = 0.123
  weight_decay = 0.456
  betas = c(0.789, 0.444)
  eps = 0.01
  amsgrad = sample(c(TRUE, FALSE), 1)
  o = make_adamw(lr = lr, weight_decay = weight_decay, betas = betas, eps = eps, amsgrad = amsgrad)
  o$state_dict()

  x = o$state_dict()

  o$load_state_dict(sd)

  o$param_groups[[1]]$lr = 10
  expect_equal(o$param_groups[[1]]$lr, 10)


  x = o$state_dict()
  o$load_state_dict(x)
  x
  o$param_groups[[1]]$lr = 0.2
  expect_equal(x$param_groups[[1]]$lr, 0.2)

  # TODO: Test that no segfaults
  # test that state_dict()$param_groups[[1]]$params is a vector of ints
  # and that they cover the whole sequence of the 1:n
  # test that step is a long
  # test that the order is preserved
  # test that we can also set the state
  # test that amsgrad is defined it it is TRUE

  o$state_dict2()
  expect_equal()
  gc(); gc()
  gs = o$state_dict2()
  gs = o$state_dict2()
  gc(); gc()
  # When we bind the param_groups to a variable, then calling gc() will not segfault
  # only when we again bind the param_groups to a variabe and call gc() will it segfault
  #
  # I think this means that the delter of the external pointer frees twice?

  gs = o$param_groups
  gs = o$param_groups
  gc()
  gc()
  print.default(o$state_dict2()[[2]]$max_exp_avg_sq)
  o$param_groups[[1]]

  expect_equal(o$param_groups[[1]]$lr, lr)
  expect_equal(o$param_groups[[1]]$weight_decay, weight_decay)
  expect_equal(o$param_groups[[1]]$betas, betas)
  expect_equal(o$param_groups[[1]]$eps, eps)
  expect_equal(o$param_groups[[1]]$amsgrad, amsgrad)
})

test_that("loading a state dict works", {
  # TODO: Focus also on the order.
})

test_that("param_groups works", {
  # TODO: Write a lot of tests
  library(ignite)
  library(torch)

  o = make_adamw(lr = 0.1, amsgrad = TRUE)
  o$state_dict2()
  o$param_groups

  ignite:::rcpp_ignite_adamw_state(o$ptr)
  # TODO: Something is off with the parameter conversion


  o$param_groups[[1]]$lr = 100
  expect_equal(o$param_groups[[1]]$lr, 100)
})

test_that("state_dict works", {
  o = make_adamw(lr = 0.1, momentum = 0.9)
  sd = o$state_dict()
  expect_equal(length(sd), 1)
  expect_equal(sd[[1]]$lr, 0.1)
})
