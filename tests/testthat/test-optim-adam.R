test_that("optim_adam", {
  expect_optim_works(optim_adam, list(lr = 0.1))
  expect_optim_works(optim_adam, list(lr = 0.1, weight_decay = 1e-5))
  expect_optim_works(optim_adam, list(lr = 0.1, weight_decay = 1e-5, amsgrad = TRUE))
  expect_state_is_updated(optim_adam)
})

test_that("optim state_dict with more parameters", {
  model <- nn_sequential(
    nn_linear(10, 10),
    nn_relu(),
    nn_linear(10, 10),
    nn_relu(),
    nn_linear(10, 1)
  )

  opt <- optim_adam(model$parameters)
  opt$zero_grad()

  input <- torch_randn(100, 10)
  target <- torch_randn(100, 1)

  loss <- nnf_mse_loss(model(input), target)
  loss$backward()

  opt$step()

  state_dict <- opt$state_dict()
  expect_length(state_dict$state, 6)
  lapply(state_dict$state, function(x) {
    expect_length(x, 3)
  })
  expect_length(state_dict$param_groups, 1)
  expect_equal(state_dict$param_groups[[1]]$params, as.numeric(1:6))

  model2 <- nn_sequential(
    nn_linear(10, 10),
    nn_relu(),
    nn_linear(10, 10),
    nn_relu(),
    nn_linear(10, 1)
  )
  opt2 <- optim_adam(model2$parameters)
  opt2$load_state_dict(state_dict)

  d <- opt2$state_dict()
  expect_equal(d, state_dict)
})

test_that("regression test for #559", {
  tmp <- tempfile()

  model <- nn_linear(1, 1)
  opt <- optim_adam(model$parameters)

  train_x <- torch_randn(100, 1)
  train_y <- torch_randn(100, 1)

  # first pass prior to saving and loading: succeeds

  loss <- nnf_mse_loss(model(train_x), train_y)
  loss$backward()
  opt$step()

  # save model
  torch_save(model, tmp)

  rm(model)
  gc()

  # second pass
  model <- torch_load(tmp)
  opt <- optim_adam(model$parameters)

  loss <- nnf_mse_loss(model(train_x), train_y)
  loss$backward()

  # where it fails
  expect_error(
    opt$step(),
    regexp = NA
  )

  expect_length(opt$state$map, 2)
})
