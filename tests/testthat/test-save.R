test_that("save tensor", {
  fname <- tempfile(fileext = "pt")
  x <- torch_randn(10, 10)
  torch_save(x, fname)
  y <- torch_load(fname)

  expect_equal_to_tensor(x, y)
})

test_that("save a module", {
  fname <- tempfile(fileext = "pt")

  Net <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 1)
      self$norm <- nn_batch_norm1d(1)
    },
    forward = function(x) {
      x <- self$linear(x)
      x <- self$norm(x)
      x
    }
  )
  net <- Net()

  torch_save(net, fname)
  reloaded_net <- torch_load(fname)
  gc()

  x <- torch_randn(100, 10)
  expect_equal_to_tensor(net(x), reloaded_net(x))
})

test_that("save more complicated module", {
  Net <- nn_module(
    "Net",
    initialize = function() {
      self$conv1 <- nn_conv2d(1, 32, 3, 1)
      self$conv2 <- nn_conv2d(32, 64, 3, 1)
      self$dropout1 <- nn_dropout(0.25)
      self$dropout2 <- nn_dropout(0.5)
      self$fc1 <- nn_linear(9216, 128)
      self$fc2 <- nn_linear(128, 10)
    },
    forward = function(x) {
      x <- self$conv1(x)
      x <- nnf_relu(x)
      x <- self$conv2(x)
      x <- nnf_relu(x)
      x <- nnf_max_pool2d(x, 2)
      x <- self$dropout1(x)
      x <- torch_flatten(x, start_dim = 2)
      x <- self$fc1(x)
      x <- nnf_relu(x)
      x <- self$dropout2(x)
      x <- self$fc2(x)
      output <- nnf_log_softmax(x, dim = 1)
      output
    }
  )
  fname <- tempfile(fileext = ".pt")

  net <- Net()


  torch_save(net, fname)
  reloaded_net <- torch_load(fname)

  gc()

  expect_equal_to_tensor(
    net$conv1$parameters$weight,
    reloaded_net$conv1$parameters$weight
  )
  expect_equal_to_tensor(
    net$conv1$parameters$bias,
    reloaded_net$conv1$parameters$bias
  )

  expect_equal_to_tensor(
    net$conv2$parameters$weight,
    reloaded_net$conv2$parameters$weight
  )
  expect_equal_to_tensor(
    net$conv2$parameters$bias,
    reloaded_net$conv2$parameters$bias
  )

  expect_equal_to_tensor(
    net$fc1$parameters$weight,
    reloaded_net$fc1$parameters$weight
  )
  expect_equal_to_tensor(
    net$fc1$parameters$bias,
    reloaded_net$fc1$parameters$bias
  )

  expect_equal_to_tensor(
    net$fc2$parameters$weight,
    reloaded_net$fc2$parameters$weight
  )
  expect_equal_to_tensor(
    net$fc2$parameters$bias,
    reloaded_net$fc2$parameters$bias
  )

  net$train(FALSE)
  reloaded_net$train(FALSE)

  x <- torch_randn(10, 1, 28, 28)
  expect_equal_to_tensor(net(x), reloaded_net(x))
})

test_that("save alexnet like model", {
  net <- nn_module(
    "Net",
    initialize = function() {
      self$features <- nn_sequential(
        nn_conv2d(3, 5, kernel_size = 11, stride = 4, padding = 2),
        nn_relu()
      )
      self$avgpool <- nn_max_pool2d(c(6, 6))
      self$classifier <- nn_sequential(
        nn_dropout(),
        nn_linear(10, 10),
        nn_relu(),
        nn_dropout()
      )
    },
    forward = function(x) {
      x <- self$features(x)
      x <- self$avgpool(x)
      x <- torch_flatten(x, start_dim = 2)
      x <- self$classifier(x)
    }
  )

  model <- net()

  fname <- tempfile(fileext = ".pt")
  torch_save(model, fname)
  m <- torch_load(fname)

  pars <- model$parameters
  r_pars <- m$parameters

  for (i in seq_along(pars)) {
    expect_equal_to_tensor(pars[[i]], r_pars[[i]])
  }
})

test_that("load a state dict created in python", {

  # the state dict was create in python with
  # ones = torch.ones(3, 5)
  # twos = torch.ones(3, 5) * 2
  # value = {'ones': ones, 'twos': twos}
  # torch.save(value, "assets/state_dict.pth", _use_new_zipfile_serialization=True)

  dict <- load_state_dict(test_path("assets/state_dict.pth"))
  expect_equal(names(dict), c("ones", "twos"))
  expect_equal_to_tensor(dict$ones, torch_ones(3, 5))
  expect_equal_to_tensor(dict$twos, torch_ones(3, 5) * 2)
})

test_that("can load a state dict that contains an ordered dict", {

  dict <- load_state_dict(test_path("assets/ordered_dict.pt"))
  expect_equal(names(dict), c("weight", "bias"))
  expect_tensor_shape(dict$weight, c(10, 10))
  expect_tensor_shape(dict$bias, c(10))
})

test_that("Can load a torch v0.2.1 model", {
  skip_on_os("windows")

  dest <- testthat::test_path("assets/model-v0.2.1.pt")
  if (!file.exists(dest)) {
    download.file(
      "https://torch-cdn.mlverse.org/testing-models/v0.2.1.pt", 
      destfile = dest, 
      mode = "wb"
    )  
  }

  model <- torch_load(dest)
  x <- torch_randn(32, 1, 28, 28)

  suppressWarnings({
    expect_error(o <- model(x), regexp = NA)  
  })
  expect_tensor_shape(o, c(32, 10))
})

test_that("Can load a v0.10.0 model", {
  
  dest <- testthat::test_path("assets/model-v0.10.0.pt")
  if (!file.exists(dest)) {
    download.file(
      "https://torch-cdn.mlverse.org/testing-models/v0.10.0.pt", 
      destfile = dest, 
      mode = "wb"
    )  
  }
  
  model <- torch_load(dest)
  x <- torch_randn(32, 1, 28, 28)
  
  suppressWarnings({
    expect_error(o <- model(x), regexp = NA)  
  })
  expect_tensor_shape(o, c(32, 10))
  
})

test_that("requires_grad for tensors is maintained", {
  x <- torch_randn(10, 10, requires_grad = TRUE)
  tmp <- tempfile("model", fileext = "pt")
  torch_save(x, tmp)
  y <- torch_load(tmp)
  expect_true(y$requires_grad)

  x <- torch_randn(10, 10, requires_grad = FALSE)
  tmp <- tempfile("model", fileext = "pt")
  torch_save(x, tmp)
  y <- torch_load(tmp)
  expect_false(y$requires_grad)
})

test_that("requires_grad of parameters is correct", {
  model <- nn_linear(10, 10)
  tmp <- tempfile("model", fileext = "pt")
  torch_save(model, tmp)
  model2 <- torch_load(tmp)
  expect_true(model2$bias$requires_grad)


  model <- nn_linear(10, 10)
  model$bias$requires_grad_(FALSE)
  expect_false(model$bias$requires_grad)
  tmp <- tempfile("model", fileext = "pt")
  torch_save(model, tmp)
  model2 <- torch_load(tmp)
  expect_false(model2$bias$requires_grad)
})

test_that("can save with a NULL device", {
  skip_if_cuda_not_available()

  model <- nn_linear(10, 10)$cuda()
  tmp <- tempfile("model", fileext = "pt")
  torch_save(model, tmp)
  
  expect_error({
    model <- torch_load(tmp, device = NULL)  
  }, "Unexpected device")
})

test_that("save on cuda and load on cpu", {
  skip_if_cuda_not_available()
  model <- nn_linear(10, 10)$cuda()

  expect_equal(model$weight$device$type, "cuda")

  tmp <- tempfile("model", fileext = "pt")
  torch_save(model, tmp)

  mod <- torch_load(tmp)

  expect_equal(mod$weight$device$type, "cpu")
})

test_that("save on cuda and load on cuda", {
  skip_if_cuda_not_available()
  model <- nn_linear(10, 10)$cuda()

  expect_equal(model$weight$device$type, "cuda")

  tmp <- tempfile("model", fileext = "pt")
  torch_save(model, tmp)

  mod <- torch_load(tmp, device = "cuda")

  expect_equal(mod$weight$device$type, "cuda")
})

test_that("can save and load from lists", {
  l <- list(
    torch_tensor(1),
    a = torch_tensor(2),
    b = list(
      x = torch_tensor(3),
      y = 4
    ),
    c = 5
  )

  tmp <- tempfile()
  torch_save(l, tmp)

  rm(l)
  gc()

  l <- torch_load(tmp)
  expect_equal_to_tensor(l[[1]], torch_tensor(1))
  expect_equal_to_tensor(l$a, torch_tensor(2))
  expect_equal_to_tensor(l$b$x, torch_tensor(3))
  expect_equal(l$b$y, 4)
  expect_equal(l$c, 5)
})

test_that("can use torch_serialize", {
  
  model <- nn_linear(10, 10)
  x <- torch_randn(10, 10)
  ser <- torch_serialize(model)
  pred <- model(x)
  
  rm(model); gc();
  
  model2 <- torch_load(ser)
  pred2 <- model2(x)
  
  expect_true(torch_allclose(pred, pred2))
  expect_error(regexp = "matched", {
    ser <- torch_serialize(model2, path = tempfile())  
  })
  
})

test_that("is_rds should't move the connection position", {
  raw <- charToRaw("hello world")
  con <- rawConnection(raw, open = "rb")
  on.exit({close(con)}, add = TRUE)
  
  check <- is_rds(con)
  expect_equal(check, FALSE)
  
  x <- readBin(raw, n = 1, character())
  expect_equal(x, "hello world")
  
  x <- torch_serialize(nn_linear(10, 10))
  expect_true(!is_rds(x))
  expect_true(inherits(torch_load(x), "nn_module"))
})

test_that("saving tensor with ser3", {
  
  x <- torch_randn(10, 10)
  tmp <- tempfile()
  withr::with_options(c(torch.serialization_version = 3), {
    torch_save(x, tmp)
  })
  y <- torch_load(tmp)
  expect_true(torch_allclose(x, y))

})

test_that("saving lists with ser3", {
  
  z <- torch_randn(10, 10)
  x <- list(x = torch_randn(10, 10, requires_grad = TRUE), torch_randn(10, 10), z, z)
  tmp <- tempfile()
  withr::with_options(c(torch.serialization_version = 3), {
    torch_save(x, tmp)
  })
  
  l <- torch_load(tmp)
  expect_equal(names(l), c("x", "", "", ""))
  expect_true(torch_allclose(l$x, x$x))
  expect_true(l$x$requires_grad)
  expect_true(torch_allclose(l[[2]], x[[2]]))
  expect_false(l[[2]]$requires_grad)
  expect_equal(xptr_address(l[[3]]), xptr_address(l[[4]]))
  
})

test_that("can save module with ser3", {
  
  module <- nn_linear(10, 10)
  tmp <- tempfile()
  withr::with_options(c(torch.serialization_version = 3), {
    torch_save(module, tmp)
  })
  
  mod <- torch_load(tmp)
  expect_true(torch_allclose(module$weight, mod$weight))
  expect_true(torch_allclose(module$bias, mod$bias))
})

test_that("can save datasets with ser3", {
  
  dt <- dataset(
    initialize = function() {
      self$x <- torch_randn(10, 10)
      self$y <- torch_randn(10, 10)
    },
    .getitem = function(i) {
      list(self$x[i,], self$y[i,])
    },
    .length = function() {
      10
    }
  )
  
  d <- dt()
  tmp <- tempfile()
  
  withr::with_options(c(torch.serialization_version = 3), {
    torch_save(d, tmp)
  })
  
  d2 <- torch_load(tmp)
  
  expect_true(torch_allclose(d$x, d2$x))
  expect_true(torch_allclose(d$y, d2$y))
  expect_true(torch_allclose(d[1][[1]], d2[1][[1]]))
})

test_that("can save a complex tensor", {
  z <- torch_randn(10, 10)$to(dtype="cfloat")
  k <- torch_serialize(z)
  x <- torch_load(k)
  
  expect_true(torch_allclose(x$real, z$real))
  expect_true(torch_allclose(x$imag, z$imag))
})