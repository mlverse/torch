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
  with_autocast(device_type="cpu", {
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
  })
  
  expect_equal(w$dtype$.type(), "BFloat16")
  expect_true(w$device == torch_device("cuda"))
  
  a <- torch_mm(x, w$float())
  expect_true(a$dtype == torch_float())
  
})

test_that("grad scalers work correctly", {
  
  skip_if_cuda_not_available()
  device <- torch_device("cuda")
  
  # Creates model and optimizer in default precision
  model <- nn_linear(10, 1)$cuda()
  optimizer <- optim_sgd(model$parameters, lr = 0.001)
  
  # Creates a GradScaler once at the beginning of training.
  scaler <- amp_GradScaler$new()
  
  for (epoch in 1:5) {
    x <- torch_randn(100, 10, device = device)
    y <- torch_randn(100, 1, device = device)
    
    with_autocast(device_type = "cuda", {
      output <- model(x)
      loss <- nnf_mse_loss(output, y)
    })
    
    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
    # Backward passes under autocast are not recommended.
    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
    scaler$scale(loss)$backward()
    
    # scaler.step() first unscales the gradients of the optimizer's assigned params.
    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
    # otherwise, optimizer.step() is skipped.
    scaler$step(optimizer)
    
    # Updates the scale for next iteration.
    scaler$update()
  }
  
  # no Inf values
  expect_true(!torch::torch_isinf(model$weight)$any()$item())
})
