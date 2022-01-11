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
      output <- nnf_log_softmax(x, dim = 1)
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
    jit_trace(fn, "a")
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

test_that("fn can take more than 1 argument", {
  fn <- function(x, y) {
    list(x, x + y)
  }

  x <- torch_tensor(1)
  y <- torch_tensor(2)

  tr_fn <- jit_trace(fn, x, y)
  expect_equal_to_tensor(fn(x, y)[[1]], tr_fn(x, y)[[1]])
  expect_equal_to_tensor(fn(x, y)[[2]], tr_fn(x, y)[[2]])

  expect_error(
    tr_fn <- jit_trace(fn, x = x, y = y)
  )
})

test_that("can have named inputs and outputs", {
  fn <- function(x) {
    list(x = x$t1, y = x$t2)
  }

  x <- list(
    t1 = torch_tensor(1),
    t2 = torch_tensor(2)
  )

  tr_fn <- jit_trace(fn, x, strict = FALSE)

  expect_equal(
    tr_fn(x),
    fn(x)
  )
})

test_that("tuple inputs are correctly handled", {
  fn <- function(x) {
    jit_tuple(list(x = x$t1, y = x$t2))
  }

  x <- jit_tuple(list(
    t1 = torch_tensor(1),
    t2 = torch_tensor(2)
  ))

  tr_fn <- jit_trace(fn, x, strict = FALSE)

  # returned named tuples will loose their names
  expect_equal(tr_fn(x), list(torch_tensor(1), torch_tensor(2)))

  # if the model has been traced with a named tuple we
  # will expect a tuple back too.
  expect_error(
    tr_fn(list(t1 = torch_tensor(1), t2 = torch_tensor(2)))
  )
})

test_that("tuple casting", {
  fn <- function(y) {
    jit_tuple(list(y[[1]], y[[2]]))
  }

  x <- jit_tuple(list(torch_tensor(1), torch_tensor(2)))
  tr_fn <- jit_trace(fn, x)

  expect_error(
    tr_fn(list(torch_tensor(1), torch_tensor(2)))
  )

  expect_equal(
    tr_fn(jit_tuple(list(torch_tensor(1), torch_tensor(2)))),
    list(torch_tensor(1), torch_tensor(2))
  )
})

test_that("trace a nn module", {
  test_module <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 10)
      self$norm <- nn_batch_norm1d(10)
      self$par <- nn_parameter(torch_tensor(2))
      self$buff <- nn_buffer(torch_randn(10, 5))
      self$constant <- 1
      self$hello <- list(torch_tensor(1), torch_tensor(2), "hello")
    },
    forward = function(x) {
      self$par * x
    },
    testing = function(x) {
      x %>%
        self$linear()
    },
    test_constant = function(x) {
      x + self$constant + self$hello[[2]]
    }
  )

  mod <- test_module()

  expect_error(
    m <- jit_trace_module(
      mod,
      forward = torch_randn(1),
      testing = list(torch_randn(10, 10)),
      test_constant = list(torch_tensor(1))
    ),
    regexp = NA
  )

  expect_length(m$parameters, 5)
  expect_length(m$buffers, 4)
  expect_length(m$modules, 3)

  expect_equal_to_tensor(m(torch_tensor(2)), torch_tensor(4))
  with_no_grad(m$par$zero_())
  expect_equal_to_tensor(m(torch_tensor(2)), torch_tensor(0))

  x <- torch_randn(10, 10)
  expect_equal_to_tensor(m$testing(x), mod$testing(x))
  with_no_grad({
    m$linear$weight$zero_()$add_(1)
    mod$linear$weight$zero_()$add_(1)
  })
  expect_equal_to_tensor(m$testing(x), mod$testing(x))

  expect_equal_to_tensor(m$test_constant(torch_tensor(2)), torch_tensor(5))
})

test_that("dont crash when gcing a method", {
  mod <- jit_trace(nn_linear(10, 10), torch_randn(10, 10))
  gc()
  forward <- mod$forward
  rm(forward)
  gc()
  gc()
  expect_error(regexp = NA, mod$forward)
})

test_that("we can save traced modules", {
  test_module <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 10)
      self$norm <- nn_batch_norm1d(10)
      self$par <- nn_parameter(torch_tensor(2))
      self$buff <- nn_buffer(torch_randn(10, 5))
      self$constant <- 1
      self$hello <- list(torch_tensor(1), torch_tensor(2), "hello")
    },
    forward = function(x) {
      self$par * x
    },
    testing = function(x) {
      x %>%
        self$linear()
    },
    test_constant = function(x) {
      x + self$constant + self$hello[[2]]
    }
  )

  mod <- test_module()
  m <- jit_trace_module(
    mod,
    forward = torch_randn(1),
    testing = list(torch_randn(10, 10)),
    test_constant = list(torch_tensor(1))
  )

  jit_save(m, "tracedmodule.pt")
  rm(m)
  gc()
  gc()

  m <- jit_load("tracedmodule.pt")

  expect_length(m$parameters, 5)
  expect_length(m$buffers, 4)
  expect_length(m$modules, 3)

  expect_equal_to_tensor(m(torch_tensor(2)), torch_tensor(4))
  with_no_grad(m$par$zero_())
  expect_equal_to_tensor(m(torch_tensor(2)), torch_tensor(0))

  x <- torch_randn(10, 10)
  expect_equal_to_tensor(m$testing(x), mod$testing(x))
  with_no_grad({
    m$linear$weight$zero_()$add_(1)
    mod$linear$weight$zero_()$add_(1)
  })
  expect_equal_to_tensor(m$testing(x), mod$testing(x))

  expect_equal_to_tensor(m$test_constant(torch_tensor(2)), torch_tensor(5))
})

test_that("trace a module", {
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
      output <- nnf_log_softmax(x, dim = 1)
      output
    }
  )

  net <- Net()
  net$eval()

  input <- torch_randn(100, 1, 28, 28)
  out <- net(input)

  tr_fn <- jit_trace(net, input)
  expect_equal_to_tensor(net(input), tr_fn(input), tolerance = 1e-6)
})

test_that("Can recover from errors in the traced method", {
  module <- nn_module(
    initialize = function() {},
    forward = function(x) {
      stop("The error abcde")
    }
  )

  expect_error(
    jit_trace(module(), torch_tensor(1)),
    regexp = ".*abcde"
  )

  expect_error(
    regexp = "You must initialize the nn_module before tracing",
    jit_trace(module, torch_tensor(1))
  )

  expect_error(
    regexp = "jit_trace needs a function or nn_module",
    jit_trace(1, torch_tensor(1))
  )
})

test_that("we get a good error message when trying to call a method from a submodule", {
  module <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 10)
    },
    forward = function(x) {
      self$linear(x)
    }
  )

  m <- jit_trace(module(), torch_randn(100, 10))

  expect_error(
    m$linear(torch_randn(10, 10)),
    regexp = "Methods from submodules of traced modules are not traced"
  )
})

test_that("errors in the tracer are correctly captured", {
  module <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 10)
    },
    forward = function(x) {
      self$linear(x)
      1
    }
  )

  expect_error(
    jit_trace(module(), torch_randn(10, 10)),
    regexp = ".*Only tensors, lists, tuples of tensors"
  )
})

test_that("we can include traced module as a submodule and trace", {
  module <- nn_module(
    initialize = function() {
      self$linear <- jit_trace(nn_linear(10, 10), torch_randn(10, 10))
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  mod <- module()

  m <- jit_trace(mod, torch_randn(10, 10))

  x <- torch_randn(10, 10)
  expect_equal_to_tensor(m(x), mod(x))
  expect_equal_to_tensor(m$linear(x), mod$linear(x))
})

test_that("can save module for mobile", {
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
      output <- nnf_log_softmax(x, dim = 1)
      output
    }
  )

  net <- Net()
  net$eval()

  input <- torch_randn(100, 1, 28, 28)
  out <- net(input)

  tr_fn <- jit_trace(net, input)

  tmp <- tempfile("tst", fileext = ".pt")
  jit_save_for_mobile(tr_fn, tmp)

  f <- jit_load(tmp)
  expect_equal_to_tensor(net(input), f(input), tol = 1e-6)
})

test_that("can save function for mobile", {
  fn <- function(x) {
    torch_relu(x)
  }

  input <- torch_tensor(c(-1, 0, 1))
  tr_fn <- jit_trace(fn, input)

  tmp <- tempfile("tst", fileext = ".pt")
  jit_save_for_mobile(tr_fn, tmp)

  f <- jit_load(tmp)
  expect_equal_to_tensor(torch_relu(input), f(input))
})
