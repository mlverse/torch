test_that("constant padding", {
  x <- torch_ones(1, 1, 1)

  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1), mode = "constant"),
    c(1, 1, 3)
  )

  x <- torch_ones(1, 1, 1, 1)
  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1, 1, 1), mode = "constant"),
    c(1, 1, 3, 3)
  )

  x <- torch_ones(1, 1, 1, 1, 1)
  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1, 1, 1, 1, 1), mode = "constant"),
    c(1, 1, 3, 3, 3)
  )
})

test_that("circular padding", {
  x <- torch_ones(1, 1, 1)

  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1), mode = "circular"),
    c(1, 1, 3)
  )

  x <- torch_ones(1, 1, 1, 1)
  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1, 1, 1), mode = "circular"),
    c(1, 1, 3, 3)
  )

  x <- torch_ones(1, 1, 1, 1, 1)
  expect_tensor_shape(
    nnf_pad(x, pad = c(1, 1, 1, 1, 1, 1), mode = "circular"),
    c(1, 1, 3, 3, 3)
  )
})
