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
  expect_equal_to_tensor(m(x), torch_ones(1,10), tolerance = 1e-5)
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

test_that("load state dict for batch norm", {
  m <- nn_batch_norm2d(10)
  s <- m$state_dict()
  s <- s[!names(s) %in% "num_batches_tracked"]
  m$load_state_dict(s)
  
  expect_length(m$state_dict(), 5)
})
