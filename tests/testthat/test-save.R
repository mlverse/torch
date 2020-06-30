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
  
  x <- torch_randn(100, 10)
  expect_equal_to_tensor(net(x), reloaded_net(x))
  
})