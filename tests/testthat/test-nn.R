context("nn")

test_that("nn_module", {
  my_net <- nn_module(
    "my_net",
    initialize = function(n_inputs, n_outputs) {
      self$W <- nn_parameter(torch_randn(n_inputs, n_outputs))
      self$b <- nn_parameter(torch_zeros(n_outputs))
    },
    forward = function(x) {
      torch_addmm(self$b, x, self$W)
    }
  )
  
  model <- my_net(1,1)
  expect_s3_class(model, "nn_module")
  expect_s3_class(model, "my_net")
  expect_length(model$parameters, 2)
  expect_tensor(model(torch_randn(10,1)))
})

test_that("nn_modules can have child modules", {
  my_net <- nn_module(
    "my_net",
    initialize = function(n_inputs, n_outputs) {
      self$linear <- nn_linear(n_inputs, n_outputs)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  
  model <- my_net(1,2)
  x <- torch_randn(1, 1)
  output <- model(x)
  
  expect_s3_class(model, "nn_module")
  expect_s3_class(model, "my_net")
  expect_length(model$parameters, 2)
  expect_tensor(output)
  expect_equal(output$dim(), 2)
  
})

test_that("nn_sequential", {
  model <- nn_sequential(
    nn_linear(10, 100),
    nn_relu(),
    nn_linear(100, 1)
  )
  
  input <- torch_randn(1000, 10)
  output <- model(input)
  
  expect_tensor(output)
  expect_s3_class(model, "nn_sequential")
  expect_s3_class(model, "nn_module")
  expect_equal(output$shape, c(1000, 1))
  expect_length(model$parameters, 4)
  
  model <- nn_sequential(
    name = "mynet",
    nn_linear(10, 100),
    nn_relu(),
    nn_linear(100, 1)
  )
  expect_s3_class(model, "mynet")
  expect_s3_class(model, "nn_module")
})

test_that("nn_module_list", {
  x <- nn_module_list(list(
    nn_linear(10, 100),
    nn_relu(),
    nn_linear(100, 10)
  ))
  
  expect_s3_class(x[[1]], "nn_linear")
  expect_s3_class(x[[2]], "nn_relu")
  expect_s3_class(x[[3]], "nn_linear")
  
  x$append(nn_relu6())
  expect_s3_class(x[[4]], "nn_relu6")
  
  x$extend(list(nn_celu(), nn_gelu()))
  expect_s3_class(x[[5]], "nn_celu")
  expect_s3_class(x[[6]], "nn_gelu")
  
  x$insert(index = 1, nn_dropout())
  expect_s3_class(x[[1]], "nn_dropout")
  
  expect_length(x, 7)
})

test_that("module_list inside a module", {
  
  my_module <- nn_module(
   initialize = function() {
     self$linears <- nn_module_list(lapply(1:10, function(x) nn_linear(10, 10)))
   },
   forward = function(x) {
    for (i in 1:length(self$linears))
      x <- self$linears[[i]](x)
    x
   }
  )
  
  m <- my_module()
  expect_length(m$parameters, 20)
  output <- m(torch_randn(5, 10))
  expect_tensor(output)
  
})

test_that("to", {
  
  net <- nn_linear(10, 10)
  net$to(dtype = torch_double())
  
  expect_true(net$weight$dtype() == torch_double())
  expect_true(net$bias$dtype() == torch_double())
  
  
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
  x <- torch_randn(10, 10)
  y <- net(x)
  r <- torch_mean(y)
  r$backward()
  
  net$to(dtype = torch_double())
  
  expect_true(net$linear$weight$dtype() == torch_double())
  expect_true(net$linear$bias$dtype() == torch_double())
  expect_true(net$norm$running_mean$dtype() == torch_double())
  expect_true(net$norm$running_var$dtype() == torch_double())
  expect_true(net$linear$weight$grad$dtype() == torch_double())
  
  skip_if_cuda_not_available()
  net$cuda()
  expect_equal(net$linear$weight$device()$type, "cuda")
  expect_equal(net$linear$bias$device()$type, "cuda")
  
  net$cpu()
  expect_equal(net$linear$weight$device()$type, "cpu")
  expect_equal(net$linear$bias$device()$type,"cpu")
  
})

test_that("state_dict for modules", {
  
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
  s <- net$state_dict()  
  s
  
  expect_length(s, 7)
  expect_equal_to_tensor(s[[1]], net$linear$weight)
  expect_equal_to_tensor(s[[2]], net$linear$bias)
  expect_equal_to_tensor(s[[5]], net$norm$running_mean)
  expect_equal_to_tensor(s[[6]], net$norm$running_var)
  
  net2 <- Net()
  net2$load_state_dict(s)
  s <- net2$state_dict()
  
  expect_length(s, 7)
  expect_equal_to_tensor(s[[1]], net$linear$weight)
  expect_equal_to_tensor(s[[2]], net$linear$bias)
  expect_equal_to_tensor(s[[5]], net$norm$running_mean)
  expect_equal_to_tensor(s[[6]], net$norm$running_var)
  
  
  s <- s[-7]
  expect_error(net2$load_state_dict(s), class = "value_error")
  
})

test_that("zero_grad", {
  
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
  
  expect_no_error(net$zero_grad())
  expect_true(is_undefined_tensor(net$linear$weight$grad))
  
  x <- torch_randn(500, 10)
  l <- torch_mean(net(x))
  l$backward()
  
  expect_false(as_array(torch_all(net$linear$weight$grad == 0)))
  expect_false(as_array(torch_all(net$linear$bias$grad == 0)))
  expect_false(as_array(torch_all(net$norm$weight$grad == 0)))
  expect_false(as_array(torch_all(net$norm$bias$grad == 0)))
  
  net$zero_grad()
  
  expect_true(as_array(torch_all(net$linear$weight$grad == 0)))
  expect_true(as_array(torch_all(net$linear$bias$grad == 0)))
  expect_true(as_array(torch_all(net$norm$weight$grad == 0)))
  expect_true(as_array(torch_all(net$norm$bias$grad == 0)))
  
})

test_that("index modules with integers", {
  
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
  
  expect_equal_to_tensor(net[[1]]$weight, net$linear$weight)
  
  net <- nn_linear(10, 10)
  
  expect_error(net[[1]], "out of bounds")
  
})

test_that("to still returns an nn_module", {
  
  x <- nn_linear(10, 10)
  y <- x$to(device ="cpu")
  
  expect_s3_class(y, "nn_module")
  
  expect_tensor_shape(y(torch_randn(10, 10)), c(10, 10))
  
})

test_that("moodule$apply", {
  
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
  zero <- function(x) {
    if (!is.null(x$weight)) {
      with_no_grad({
        x$weight$zero_()
      })  
    }
  }
  net$apply(zero)
  
  expect_equal_to_tensor(net$linear$weight, torch_zeros_like(net$linear$weight))
  expect_equal_to_tensor(net$norm$weight, torch_zeros_like(net$norm$weight))
  
})
