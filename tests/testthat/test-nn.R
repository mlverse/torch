test_that("nn_module", {
  Net <- nn_module(
    initialize = function(n_inputs, n_outputs) {
      self$W <- nn_parameter(torch_randn(n_inputs, n_outputs))
      self$b <- nn_parameter(torch_zeros(n_outputs))
    },
    forward = function(x) {
      torch_addmm(self$b, x, self$W)
    }
  )
  
  model <- Net$new(1,1)
  expect_s3_class(model, "nn_module")
  expect_length(model$parameters, 2)
  expect_tensor(model$forward(torch_randn(10,1)))
})
