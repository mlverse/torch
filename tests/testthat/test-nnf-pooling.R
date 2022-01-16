test_that("1d pooling", {
  pools <- list(
    nnf_max_pool1d,
    nnf_avg_pool1d
  )

  x <- torch_randn(100, 10, 5)
  for (p in pools) {
    e <- p(x, kernel_size = 3)
    expect_tensor_shape(e, c(100, 10, 1))

    e <- p(x, kernel_size = 3, stride = 2)
    expect_tensor_shape(e, c(100, 10, 2))
  }

  e <- nnf_lp_pool1d(x, norm_type = 2, kernel_size = 3)
  expect_tensor_shape(e, c(100, 10, 1))

  e <- nnf_adaptive_avg_pool1d(x, output_size = 1)
  expect_tensor_shape(e, c(100, 10, 1))

  e <- nnf_adaptive_max_pool1d(x, output_size = 1)
  expect_tensor_shape(e, c(100, 10, 1))
})

test_that("2d pooling", {
  pools <- list(
    nnf_max_pool2d,
    nnf_avg_pool2d
  )

  x <- torch_randn(100, 10, 5, 5)
  for (p in pools) {
    e <- p(x, kernel_size = 3)
    expect_tensor_shape(e, c(100, 10, 1, 1))

    e <- p(x, kernel_size = 3, stride = 2)
    expect_tensor_shape(e, c(100, 10, 2, 2))
  }

  e <- nnf_lp_pool2d(x, norm_type = 2, kernel_size = 3)
  expect_tensor_shape(e, c(100, 10, 1, 1))

  e <- nnf_adaptive_avg_pool2d(x, output_size = c(1, 1))
  expect_tensor_shape(e, c(100, 10, 1, 1))

  e <- nnf_adaptive_max_pool2d(x, output_size = c(1, 1))
  expect_tensor_shape(e, c(100, 10, 1, 1))
})
