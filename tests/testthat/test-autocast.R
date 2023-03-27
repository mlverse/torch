test_that("local_autocast works", {
  
  x <- torch_randn(5, 5, dtype = torch_float32())
  y <- torch_randn(5, 5, dtype = torch_float32())
  
  foo <- function(x, y) {
    local_autocast(device = "cpu")
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
    w
  }
  
  out <- foo(x, y)
  expect_equal(out$dtype$.type(), "BFloat16")
  
  a <- torch_mm(x, out$float())
  expect_true(a$dtype == torch_float())
  
})

test_that("with autocast works", {
  
  x <- torch_randn(5, 5, dtype = torch_float32())
  y <- torch_randn(5, 5, dtype = torch_float32())
  with_autocast(device_type="cpu", {
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
  })
  
  expect_equal(w$dtype$.type(), "BFloat16")
  a <- torch_mm(x, w$float())
  expect_true(a$dtype == torch_float())
  
})

test_that("works on gpu", {
  
  skip_if_cuda_not_available()
  
  x <- torch_randn(5, 5, dtype = torch_float32(), device="cuda")
  y <- torch_randn(5, 5, dtype = torch_float32(), device="cuda")
  with_autocast(device_type="cuda", {
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
  })
  
  expect_equal(w$dtype$.type(), "Half")
  expect_true(w$device == torch_device("cuda", 0))
  
  a <- torch_mm(x, w$float())
  expect_true(a$dtype == torch_float())
  
})

test_that("unscale skipping works", {
  model <- nn_linear(2, 2)$cuda()
  x <- lapply(1:50, function(x) torch_randn(2, 2, dtype = torch_float32(), device="cuda"))
  y <- lapply(1:50, function(x) torch_randn(2, 2, dtype = torch_float32(), device="cuda"))
  loss_fn <- nn_mse_loss()

  orig_params <- lapply(model$parameters, function(x) x$clone()$detach())

  optimizer <- optim_sgd(model$parameters, lr=0.001)
  scaler <- cuda_amp_grad_scaler(enabled=TRUE, init_scale=128.0)
  for(i in seq_along(x)) {  
    with_autocast(device_type="cuda", dtype=torch_float16(), {
      output <- model(x[[i]])
      loss <- loss_fn(output, y[[i]])
    })
    scaler$scale(loss)$backward()
    scaler$unscale_(optimizer)
    
    # deliberately break grads
    model$parameters[[1]]$grad$copy_(torch_tensor(Inf)$cuda())
    model$parameters[[2]]$grad$copy_(torch_tensor(NaN)$cuda())
    
    scaler$step(optimizer)
    scaler$update()
  }

  expect_equal_to_tensor(model$parameters[[1]]$cpu(), orig_params[[1]]$cpu())
  expect_equal_to_tensor(model$parameters[[2]]$cpu(), orig_params[[2]]$cpu())
})

test_that("loss is scaled correctly", {

  skip_if_cuda_not_available()

  model <- nn_linear(2, 2)$cuda()
  x <- torch_randn(2, 2, device="cuda")
  y <- torch_randn(2, 2, device="cuda")

  loss_fn <- nn_mse_loss()$cuda()

  scaler <- cuda_amp_grad_scaler(init_scale = 1000)
  with_autocast(
    device_type="cuda",
    dtype=torch_float16(),
    {
      output <- model(x)
      loss <- loss_fn(output, y)
    }
  )
  scaled_loss <- scaler$scale(loss)
  expect_equal((scaled_loss/loss)$item(), scaler$.scale$item())

})

test_that("scaling the loss works", {

  model <- nn_linear(2, 2)$cuda()
  for(par in model$parameters) {
    # initialize parameters with 0 so gradients should also be small
    nn_init_constant_(par, 0) 
  }
  x <- torch_randn(2048, 2, device="cuda")/1e3
  y <- torch_randn(2048, 2, device="cuda")/1e3

  loss_fn <- nn_mse_loss()$cuda()

  with_autocast(
    device_type="cuda",
    dtype=torch_float16(),
    {
      output <- model(x)
      loss <- loss_fn(output, y)
    }
  )
  loss$backward()
  
  # gradients are so small that they become 0
  expect_true(all(as.matrix(model$weight$grad$cpu()) == 0))

  # now we scale the loss and gradients
  scaler <- cuda_amp_grad_scaler()
  with_autocast(
    device_type="cuda",
    dtype=torch_float16(),
    {
      output <- model(x)
      loss <- loss_fn(output, y)
    }
  )
  scaler$scale(loss)$backward()
  model$weight$grad
  expect_true(!any(as.matrix(model$weight$grad$cpu()) == 0))

})

test_that("internal cpp_amp_check works", {

  net <- nn_linear(2, 2)$cuda()
  x <- torch_randn(2, 2, device="cuda")
  y <- torch_randn(2, 2, device="cuda")
  loss_fn <- nn_mse_loss()$cuda()
  loss <- loss_fn(net(x), y)
  loss$backward()

  dummy_found_inf <- torch_full(list(), 0, device="cuda")
  inv_scale <- torch_full(list(), 1, device="cuda")
  found_inf <- cpp_amp_foreach_non_finite_check_and_unscale(net$parameters, dummy_found_inf, inv_scale)

  expect_equal(found_inf, 0)

  net$weight$grad$copy_(torch_tensor(Inf)$cuda())
  found_inf <- cpp_amp_foreach_non_finite_check_and_unscale(net$parameters, dummy_found_inf, inv_scale)
  expect_equal(found_inf, 1)

  net$bias$grad$copy_(torch_tensor(NaN)$cuda())
  found_inf <- cpp_amp_foreach_non_finite_check_and_unscale(net$parameters, dummy_found_inf, inv_scale)
  expect_equal(found_inf, 2)

})

test_that("grad scalers work correctly", {
  
  skip_if_cuda_not_available()

  make_model <- function(in_size, out_size, num_layers) {
    layers <- list()
    for (i in seq_len(num_layers-1)) {
      layers <- c(layers, list(nn_linear(in_size, in_size), nn_relu()))
    }
    layers <- c(layers, list(nn_linear(in_size, out_size)))
    nn_sequential(!!!layers)$cuda()
  }

  torch_manual_seed(1)

  batch_size = 512 # Try, for example, 128, 256, 513.
  in_size = 4096
  out_size = 4096
  num_layers = 3
  num_batches = 50
  epochs = 3

  # Creates data in default precision.
  # The same data is used for both default and mixed precision trials below.
  # You don't need to manually change inputs' dtype when enabling mixed precision.
  data <- lapply(1:num_batches, function(x) torch_randn(batch_size, in_size, device="cuda"))
  targets <- lapply(1:num_batches, function(x) torch_randn(batch_size, in_size, device="cuda"))

  loss_fn <- nn_mse_loss()$cuda()
  
  use_amp <- TRUE
  use_scaling <- TRUE

  net <- make_model(in_size, out_size, num_layers)
  opt <- optim_sgd(net$parameters, lr=0.1)
  scaler <- cuda_amp_grad_scaler(enabled=use_scaling)

  for (epoch in seq_len(epochs)) {
    for (i in seq_along(data)) {
      with_autocast(device_type="cuda", enabled=use_amp, {
        output <- net(data[[i]])
        loss <- loss_fn(output, targets[[i]])
      })
      scaled_loss <- scaler$scale(loss)
      scaled_loss$backward()
      
      scaler$step(opt)
      scaler$update()
      opt$zero_grad()
    }
  }

  # got the same value as obtained from pytorch
  expect_equal(
    sprintf("%1.6f", loss$item()),
    sprintf("%1.6f", 1.00434148311615)
  )
})
