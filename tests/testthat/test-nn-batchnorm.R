test_that("nn_batch_norm1d", {
  x <- torch_randn(10, 10)
  m <- nn_batch_norm1d(10, 1)
  expect_tensor_shape(m(x), c(10, 10))

  x <- torch_randn(10, 10, 10)
  m <- nn_batch_norm1d(10, 1)
  expect_tensor_shape(m(x), c(10, 10, 10))

  x <- torch_randn(10, 10, 10, 10)
  expect_error(m(x))

  x <- torch_ones(1, 10)
  m <- nn_batch_norm1d(10)
  m$eval()
  expect_equal_to_tensor(m(x), torch_ones(1, 10), tolerance = 1e-5)

  x <- torch_ones(2, 1)
  m <- nn_batch_norm1d(1)
  for (i in 1:10) {
    y <- m(x)
  }
  expect_equal_to_tensor(m$running_mean, torch_tensor(0.6513), tolerance = 1e-4)
  expect_equal_to_tensor(m$running_var, torch_tensor(0.3487), tolerance = 1e-4)
  m$eval()
  expect_equal_to_tensor(m(x), torch_empty(2, 1)$fill_(0.5905), tolerance = 1e-4)
})

test_that("nn_batch_norm2d", {
  m <- nn_batch_norm2d(10, 1)
  x <- torch_randn(10, 10, 10, 10)
  expect_tensor_shape(m(x), c(10, 10, 10, 10))

  x <- torch_randn(10, 10, 10)
  expect_error(m(x))

  m <- nn_batch_norm2d(10, 1)
  x <- torch_ones(10, 10, 10, 10)
  m$eval()
  expect_equal_to_tensor(m(x), torch_empty(10, 10, 10, 10)$fill_(0.7071), tolerance = 1e-5)
})

test_that("nn_batch_norm3d", {
  m <- nn_batch_norm3d(10, 1)
  x <- torch_randn(10, 10, 10, 10, 10)
  expect_tensor_shape(m(x), c(10, 10, 10, 10, 10))

  x <- torch_randn(10, 10, 10, 10)
  expect_error(m(x))

  m <- nn_batch_norm3d(10, 1)
  x <- torch_ones(10, 10, 10, 10, 10)
  m$eval()
  expect_equal_to_tensor(m(x), torch_empty(10, 10, 10, 10, 10)$fill_(0.7071), tolerance = 1e-5)
})

test_that("load state dict for batch norm", {
  m <- nn_batch_norm2d(10)
  s <- m$state_dict()
  s <- s[!names(s) %in% "num_batches_tracked"]
  m$load_state_dict(s)

  expect_length(m$state_dict(), 5)
})
