test_that("nn_batch_norm1d", {
  x <- torch_randn(10, 10)
  m <- nn_batch_norm1d(10, 1)
  expect_tensor_shape(m(x), c(10, 10))
  
  x <- torch_randn(10, 10, 10)
  m <- nn_batch_norm1d(10, 1)
  expect_tensor_shape(m(x), c(10, 10, 10))
  
  x <- torch_randn(10, 10, 10, 10)
  expect_error(m(x))
})

test_that("nn_batch_norm2d", {
  m <- nn_batch_norm2d(10, 1)
  x <- torch_randn(10, 10, 10, 10)
  expect_tensor_shape(m(x), c(10, 10, 10, 10))
  
  x <- torch_randn(10, 10, 10)
  expect_error(m(x))
})

test_that("load state dict for batch norm", {
  m <- nn_batch_norm2d(10)
  s <- m$state_dict()
  s <- s[!names(s) %in% "num_batches_tracked"]
  m$load_state_dict(s)
  
  expect_length(m$state_dict(), 5)
})
