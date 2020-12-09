test_that("simnple tracing works", {
  
  fn <- function(x) {
    torch_relu(x)
  }
  
  input <- torch_tensor(c(-1, 0, 1))
  tr_fn <- jit_trace(fn, input)
  
  expect_equal_to_tensor(tr_fn(input), fn(input))
})

test_that("print the graph works", {
  
  fn <- function(x) {
    torch_relu(x)
  }
  
  input <- torch_tensor(c(-1, 0, 1))
  tr_fn <- jit_trace(fn, input)
  
  expect_output(print(tr_fn$graph), regexp = "graph")
})

test_that("modules are equivalent", {
  
  Net <- nn_module(
    "Net",
    initialize = function() {
      self$conv1 <- nn_conv2d(1, 32, 3, 1)
      self$conv2 <- nn_conv2d(32, 64, 3, 1)
      self$dropout1 <- nn_dropout2d(0.25)
      self$dropout2 <- nn_dropout2d(0.5)
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
      output <- nnf_log_softmax(x, dim=1)
      output
    }
  )
    
  net <- Net()
  net$eval()
  
  # currently we need to detach all parameters in order to
  # JIT compile. We need to support modules to avoid that.
  for (p in net$parameters) {
    p$detach_()
  }
  
  fn <- function(x) {
    net(x)
  }
  
  input <- torch_randn(100, 1, 28, 28)
  out <- fn(input)
  
  tr_fn <- jit_trace(fn, input)
  expect_equal_to_tensor(fn(input), tr_fn(input), tolerance = 1e-6)
})

test_that("can save and reload", {
  
  fn <- function(x) {
    torch_relu(x)
  }
  
  input <- torch_tensor(c(-1, 0, 1))
  tr_fn <- jit_trace(fn, input)
  
  tmp <- tempfile("tst", fileext = "pt")
  jit_save(tr_fn, tmp)
  
  f <- jit_load(tmp)
  expect_equal_to_tensor(f(input), fn(input))
  
})

test_that("errors gracefully when passing unsupported inputs", {
  
  fn <- function(x) {
    torch_relu(x)
  }
  
  expect_error(
    jit_trace(fn, "a"),
    class = "runtime_error",
    regexp = "Unsupported"
  )
  
})

test_that("can take lists of tensors as input", {
  
  fn <- function(x) {
    torch_stack(x)
  }
  x <- list(torch_tensor(1), torch_tensor(2))
  
  tr_fn <- jit_trace(fn, x)
  expect_equal_to_tensor(fn(x), tr_fn(x))
  
})

test_that("can output a list of tensors", {
  
  fn <- function(x) {
    list(x, x + 1)
  }
  x <- torch_tensor(1)
  tr_fn <- jit_trace(fn, x)  
  expect_equal_to_tensor(fn(x)[[1]], tr_fn(x)[[1]])
  expect_equal_to_tensor(fn(x)[[2]], tr_fn(x)[[2]])
  
})
