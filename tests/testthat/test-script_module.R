test_that("script module parameters", {
  script_module <- jit_load("assets/linear.pt")
  parameters <- script_module$parameters

  expect_equal(names(parameters), c("weight", "bias"))
  expect_tensor_shape(parameters$weight, c(10, 10))
  expect_tensor_shape(parameters$bias, c(10))
})

test_that("script_module is callable", {
  script_module <- jit_load("assets/linear.pt")
  tensor <- script_module(torch_randn(100, 10))

  expect_tensor(tensor)
})

test_that("parameters are modifiable in-place", {
  script_module <- jit_load("assets/linear.pt")
  parameters <- script_module$parameters

  with_no_grad({
    parameters$weight$zero_()
  })

  parameters <- script_module$parameters
  expect_equal_to_tensor(parameters$weight, torch_zeros(10, 10))
})

test_that("train works", {
  script_module <- jit_load("assets/linear.pt")

  script_module$train(TRUE)
  expect_true(script_module$training)

  script_module$train(FALSE)
  expect_true(!script_module$training)

  script_module$train(TRUE)
  expect_true(script_module$training)
})

test_that("can register parameters", {
  script_module <- jit_load("assets/linear.pt")
  x <- torch_tensor(1)
  script_module$register_parameter("hello", x)
  parameters <- script_module$parameters
  expect_equal(names(parameters), c("weight", "bias", "hello"))
})

test_that("can register buffers", {
  script_module <- jit_load("assets/linear.pt")
  buffers <- script_module$buffers

  expect_length(buffers, 0)

  script_module$register_buffer("hello", torch_tensor(1))
  buffers <- script_module$buffers

  expect_length(buffers, 1)
  expect_equal(names(buffers), "hello")
  expect_equal_to_tensor(buffers[[1]], torch_tensor(1))
})

test_that("can move to device", {
  skip_if_cuda_not_available()
  script_module <- jit_load("assets/linear.pt")
  script_module$to(device = "cuda")
  parameters <- script_module$parameters

  expect_true(parameters$weight$device$type == "cuda")
  expect_true(parameters$bias$device$type == "cuda")
})

test_that("can retrieve modules", {
  script_module <- jit_load("assets/linear.pt")
  modules <- script_module$modules

  expect_length(modules, 1)

  x <- torch_randn(10, 10)
  tensor <- modules[[1]](x)

  expect_equal_to_tensor(tensor, script_module(x))
})

test_that("can apply functions", {
  script_module <- jit_load("assets/linear.pt")
  script_module$.apply(function(x) x$zero_())

  lapply(
    script_module$parameters,
    function(x) {
      expect_equal_to_tensor(x, torch_zeros_like(x))
    }
  )
})

test_that("can get the state dict and reload", {
  script_module <- jit_load("assets/linear.pt")
  state_dict <- script_module$state_dict()

  expect_length(state_dict, 2)
  state_dict[[1]] <- torch_zeros_like(state_dict[[1]])

  script_module$load_state_dict(state_dict)

  expect_equal_to_tensor(script_module$parameters[[1]], state_dict[[1]])
})

test_that("can print the graph", {
  testthat::local_edition(3)
  set.seed(1)
  traced <- jit_trace(nn_linear(10, 10), torch_randn(10, 10))

  expect_snapshot_output({
    print(traced$forward$graph)
  })

  expect_snapshot_output({
    print(traced$graph)
  })
})

test_that("graph_for", {
  testthat::local_edition(3)

  traced <- jit_trace(nn_linear(10, 10), torch_randn(10, 10))
  expect_snapshot_output({
    traced$forward$graph_for(torch_randn(10, 10))
  })

  expect_snapshot_output({
    traced$graph_for(torch_randn(10, 10))
  })
})
