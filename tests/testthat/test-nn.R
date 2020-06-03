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
  expect_length(model$parameters, 2)
  expect_tensor(model(torch_randn(10,1)))
})
