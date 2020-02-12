test_that("__and__", {
  x <- torch_tensor(TRUE)
  expect_equal_to_tensor(torch___and__(x, x), x)
  expect_equal_to_tensor(torch___and__(x, TRUE), x)
})

test_that("__lshift__", {
  x <- torch_tensor(1)
  expect_equal_to_tensor(torch___lshift__(x, x), torch_tensor(2))
  expect_equal_to_tensor(torch___lshift__(x, 1), torch_tensor(2))
})

test_that("__or__", {
  x <- torch_tensor(TRUE)
  expect_equal_to_tensor(torch___or__(x, x), x)
  expect_equal_to_tensor(torch___or__(x, FALSE), x)
})

test_that("__rshift__", {
  x <- torch_tensor(1)
  expect_equal_to_tensor(torch___rshift__(x, x), torch_tensor(0.5))
  expect_equal_to_tensor(torch___rshift__(x, 1), torch_tensor(0.5))
})

test_that("__xor__", {
  x <- torch_tensor(TRUE)
  expect_equal_to_tensor(torch___xor__(x, x), torch_tensor(FALSE))
  expect_equal_to_tensor(torch___xor__(x, TRUE), torch_tensor(FALSE))
})

test_that("_adaptive_avg_pool2d", {
  x <- torch_rand(c(10,10, 10))
  expect_tensor(torch_adaptive_avg_pool2d(x, c(2,2)))
})

test_that("_adaptive_avg_pool2d_backward", {
  x <- torch_rand(c(10,10, 10))
  b <- torch_adaptive_avg_pool2d(x, c(2,2))
  expect_tensor(torch__adaptive_avg_pool2d_backward(b, x))
})

test_that("_addr", {
  x <- torch_rand(c(2))
  y <- torch_rand(c(2))
  z <- torch_rand(c(2))
  expect_tensor(torch__addr(x, y, z))
})

test_that("_addr_", {
  x <- torch_rand(c(2,2))
  y <- torch_rand(c(2))
  z <- torch_rand(c(2))
  k <- torch__addr_(x, y, z)
  expect_equal_to_tensor(x, k)
})

test_that("_addr_out", {
  out <- torch_zeros(c(2,2))
  x <- torch_rand(c(2,2))
  y <- torch_rand(c(2))
  z <- torch_rand(c(2))
  k <- torch_addr_out(out, x, y, z)
  expect_equal_to_tensor(out, k)
})

test_that("_baddbmm_mkl_", {
  x <- torch_rand(c(2,2,2))
  y <- torch_rand(c(2,2,2))
  z <- torch_rand(c(2,2,2))
  expect_tensor(torch__baddbmm_mkl_(x, y, z))
})

test_that("_batch_norm_impl_index", {
  v <- torch_rand(c(2,2))
  w <- torch_rand(c(2))
  x <- torch_rand(c(2))
  y <- torch_rand(c(2))
  z <- torch_rand(c(2))
  out <- torch__batch_norm_impl_index(v,w,x,y,z,TRUE,1,1,TRUE)
  #TODO List returns are not correctly converted
})




