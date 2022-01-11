context("nn-linear")

test_that("nn_linear", {
  linear <- nn_linear(10, 1)
  x <- torch_randn(10, 10)

  y <- linear(x)

  expect_tensor(y)
  expect_length(as_array(y), 10)
})

test_that("initialization is identical to pytorch", {
  torch_manual_seed(1)

  expect_equal(
    nn_linear(1, 1)$weight$item(),
    0.5152631998062134 # grabbed from pytorch
  )
})
