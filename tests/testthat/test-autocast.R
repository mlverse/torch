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

  make_model <- function(in_size, out_size, num_layers) {
    layers <- list()
    for (i in 1:(num_layers - 1)) {
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
  targets <- lapply(1:num_batches, function(x) torch_randn(batch_size, out_size, device="cuda"))

  loss_fn <- nn_mse_loss()$cuda()
  
  use_amp = TRUE

  net = make_model(in_size, out_size, num_layers)
  opt = optim_sgd(net$parameters, lr=0.001)
  scaler = amp_GradScaler$new(enabled=use_amp)

  for (epoch in seq_len(epochs)) {
    for (i in length(data)) {
      with_autocast(device_type="cuda", dtype=torch_float16(), enabled=use_amp, {
        output <- net(data[[i]])
        loss <- loss_fn(output, targets[[i]])
      })
      scaler$scale(loss)$backward()
      scaler$step(opt)
      scaler$update()
      opt$zero_grad() # set_to_none=TRUE here can modestly improve performance
    }
  }

  # got the same value as obtained from pytorch
  expect_equal(
    sprintf("%1.4f", loss$item()),
    sprintf("%1.4f", 1.0086909532546997)
  )
})
