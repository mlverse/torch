context("gen-namespace")

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
  x <- torch_rand(c(10, 10, 10))
  expect_tensor(torch_adaptive_avg_pool2d(x, c(2, 2)))
})

test_that("_adaptive_avg_pool2d_backward", {
  x <- torch_rand(c(10, 10, 10))
  b <- torch_adaptive_avg_pool2d(x, c(2, 2))
  expect_tensor(torch__adaptive_avg_pool2d_backward(b, x))
})

test_that("_addr_out", {
  out <- torch_zeros(c(2, 2))
  x <- torch_rand(c(2, 2))
  y <- torch_rand(c(2))
  z <- torch_rand(c(2))
  k <- torch_addr_out(out, x, y, z)
  expect_equal_to_tensor(out, k)
})

test_that("_baddbmm_mkl_", {
  x <- torch_rand(c(2, 2, 2))
  y <- torch_rand(c(2, 2, 2))
  z <- torch_rand(c(2, 2, 2))
  expect_tensor(torch__baddbmm_mkl_(x, y, z))
})

test_that("_batch_norm_impl_index", {
  a <- torch_rand(c(2, 2))
  b <- torch_rand(c(2))
  out <- torch__batch_norm_impl_index(a, b, b, b, b, TRUE, 1, 1, TRUE)
  expect_tensor(out[[1]])
  expect_tensor(out[[2]])
  expect_tensor(out[[3]])
  expect_tensor(out[[4]])
  expect_equal(out[[5]], 0L)
})

test_that("_batch_norm_impl_index_backward", {
  skip("TODO: seems to be GPU only")
  a <- torch_rand(c(2, 2))
  b <- torch_rand(c(2))
  torch__batch_norm_impl_index_backward(1L, a, b, b, b, b, b, b, TRUE, 0.1, c(TRUE, TRUE, TRUE), b)
})

test_that("_cast_Byte", {
  x <- torch_rand(1)
  expect_tensor(torch__cast_Byte(x))
})

test_that("_cast_Char", {
  skip("TODO: Cast to characters doesn't seem to work correctly.")
  x <- torch_tensor(1234567)
  expect_tensor(torch__cast_Char(x))
})

test_that("_cast_Double", {
  x <- torch_tensor(1L, dtype = torch_int())
  expect_tensor(torch__cast_Double(x))
})

test_that("_cast_Float", {
  x <- torch_tensor(1L, dtype = torch_int())
  expect_tensor(torch__cast_Float(x))
})

test_that("_cast_Half", {
  skip("TODO: implement convertions for Half types.")
  x <- torch_tensor(1L, dtype = torch_int())
  expect_tensor(torch__cast_Half(x))
})

test_that("_cast_Int", {
  x <- torch_tensor(1)
  expect_tensor(torch__cast_Int(x))
})

test_that("_cast_Long", {
  skip("TODO: implement convertions for Long types.")
  x <- torch_tensor(1)
  expect_tensor(torch__cast_Long(x))
})

test_that("_cast_Short", {
  skip("TODO: implement convertions for Short types.")
  x <- torch_tensor(1)
  expect_tensor(torch__cast_Short(x))
})

test_that("_cat", {
  x <- torch_tensor(1)
  expect_tensor(torch__cat(list(x, x)))
})

test_that("_cat_out", {
  x <- torch_tensor(1)
  y <- torch_zeros(2)
  expect_tensor(torch__cat_out(y, list(x, x)))
  expect_equal_to_tensor(y, torch_tensor(c(1, 1)))
})

test_that("_cdist_backward", {
  x <- torch_rand(c(2, 2))
  expect_tensor(torch__cdist_backward(x, x, x, 1, x))
})

test_that("_cholesky_solve_helper", {
  x <- torch_tensor(matrix(c(1, 0, 0, 1), ncol = 2))
  expect_tensor(torch__cholesky_solve_helper(x, x, TRUE))
})

test_that("diff works", {
  a <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_diff(a), c(1, 1))

  b <- torch_tensor(c(4, 5))
  expect_equal_to_r(torch_diff(a, append = b), rep(1, 4))

  c <- torch_tensor(rbind(c(1, 2, 3), c(3, 4, 5)))
  expect_equal_to_tensor(torch_diff(c, dim = 1), torch_ones(1, 3) * 2)
  expect_equal_to_tensor(torch_diff(c, dim = 2), torch_ones(2, 2))
})

test_that("einsum", {
  x <- torch_tensor(c(1, 2, 3))
  y <- torch_tensor(c(1, 2))
  out <- torch_einsum("i,j->ij", list(x, y))
  expect_equal_to_r(out, matrix(c(1, 2, 3, 2, 4, 6), ncol = 2))
})

test_that("index", {
  x <- torch_randn(4, 4)
  y <- torch_index(x, list(torch_tensor(1:2), torch_tensor(1:2)))

  expect_equal_to_tensor(y, torch_stack(list(x[1, 1], x[2, 2])))
})

test_that("index_put", {
  x <- torch_randn(4, 4)
  y <- torch_index_put(x, list(torch_tensor(1:2), torch_tensor(1:2)), torch_tensor(0))

  x[1, 1] <- 0
  x[2, 2] <- 0

  expect_equal_to_tensor(y, x)
})

test_that("index_put_", {
  x <- torch_randn(4, 4)
  torch_index_put_(x, list(torch_tensor(1:2), torch_tensor(1:2)), torch_tensor(0))

  expect_equal(x[1, 1]$item(), 0)
  expect_equal(x[2, 2]$item(), 0)
})

test_that("logit works", {
  x <- torch_tensor(c(0.5, 0.1, 0.9))
  expect_equal_to_tensor(
    exp(torch_logit(x)) / (1 + exp(torch_logit(x))),
    x,
    tol = 1e-6
  )
})

test_that("tensordot", {
  a <- torch_arange(start = 1, end = 60)$reshape(c(3, 4, 5))
  b <- torch_arange(start = 1, end = 24)$reshape(c(4, 3, 2))
  out <- torch_tensordot(a, b, list(c(2, 1), c(1, 2)))

  expect_tensor_shape(out, c(5, 2))
})

test_that("upsample_nearest1d_out", {
  x <- torch_rand(c(2, 2, 2))
  y <- torch_zeros(c(2, 2, 2))
  expect_tensor(torch_upsample_nearest1d_out(y, x, output_size = c(2)))
  expect_not_equal_to_tensor(y, torch_zeros(c(2, 2, 2)))
})

test_that("upsample_nearest1d", {
  x <- torch_rand(c(2, 2, 2))
  expect_tensor(torch_upsample_nearest1d(x, output_size = c(2)))
})

test_that("upsample_nearest1d_backward_out", {
  x <- torch_rand(c(2, 2, 2))
  y <- torch_zeros(c(2, 2, 2))
  expect_tensor(torch_upsample_nearest1d_backward_out(y, x, c(2), c(2, 2, 2)))
  expect_not_equal_to_tensor(y, torch_zeros(c(2, 2, 2)))
})

test_that("upsample_nearest1d_backward", {
  x <- torch_rand(c(2, 2, 2))
  expect_tensor(torch_upsample_nearest1d_backward(x, c(2), c(2, 2, 2)))
})

test_that("upsample_nearest2d_out", {
  x <- torch_rand(c(2, 2, 2, 2))
  y <- torch_zeros(c(2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest2d_out(y, x, output_size = c(2, 2)))
  expect_not_equal_to_tensor(y, torch_zeros(c(2, 2, 2, 2)))
})

test_that("upsample_nearest2d", {
  x <- torch_rand(c(2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest2d(x, output_size = c(2, 2)))
})

test_that("upsample_nearest2d_backward_out", {
  x <- torch_rand(c(2, 2, 2, 2))
  y <- torch_zeros(c(2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest2d_backward_out(y, x, c(2, 2), c(2, 2, 2, 2)))
  expect_not_equal_to_tensor(y, torch_zeros(c(2, 2, 2, 2)))
})

test_that("upsample_nearest2d_backward", {
  x <- torch_rand(c(2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest2d_backward(x, c(2, 2), c(2, 2, 2, 2)))
})

test_that("upsample_nearest3d_out", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  y <- torch_zeros(c(2, 2, 2, 2, 2))
  torch_upsample_nearest3d_out(y, x, output_size = c(2, 2, 2))
  expect_tensor(y)
  expect_not_equal_to_tensor(y, torch_zeros(c(2, 2, 2, 2, 2)))
})

test_that("upsample_nearest3d", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest3d(x, output_size = c(2, 2, 2)))
})

test_that("upsample_nearest3d_backward", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  expect_tensor(torch_upsample_nearest3d_backward(x, c(2, 2, 2), c(2, 2, 2, 2, 2)))
})

test_that("upsample_nearest3d_backward_out", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  y <- torch_zeros(1)
  expect_tensor(torch_upsample_nearest3d_backward_out(y, x, c(2, 2, 2), c(2, 2, 2, 2, 2)))
})

test_that("upsample_nearest3d_out", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  y <- torch_rand(c(2, 2))
  expect_tensor(torch_upsample_nearest3d_out(y, x, c(2, 2, 2)))
  expect_tensor(y)
})

test_that("upsample_trilinear3d", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  expect_tensor(torch_upsample_trilinear3d(x, output_size = c(2, 2, 2), align_corners = TRUE))
})

test_that("upsample_trilinear3d_backward", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  expect_tensor(torch_upsample_trilinear3d_backward(x, c(2, 2, 2), c(2, 2, 2, 2, 2), align_corners = TRUE))
})

test_that("upsample_trilinear3d_backward_out", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  y <- torch_zeros(1)
  expect_tensor(torch_upsample_trilinear3d_backward_out(y, x, c(2, 2, 2), c(2, 2, 2, 2, 2), align_corners = TRUE))
})

test_that("upsample_trilinear3d_out", {
  x <- torch_rand(c(2, 2, 2, 2, 2))
  y <- torch_rand(c(2, 2))
  expect_tensor(torch_upsample_trilinear3d_out(y, x, c(2, 2, 2), align_corners = TRUE))
  expect_tensor(y)
})

test_that("var", {
  x <- torch_rand(100, names = "a")
  expect_tensor(torch_var(x))
  expect_tensor(torch_var(x, dim = 1))
  expect_tensor(torch_var(x, dim = "a"))
})

test_that("var_mean", {
  x <- torch_rand(100, names = "a")
  lapply(torch_var_mean(x), expect_tensor)
  lapply(torch_var_mean(x, dim = 1), expect_tensor)
  lapply(torch_var_mean(x, dim = "a"), expect_tensor)
})

test_that("var_out", {
  skip("TODO: see https://github.com/pytorch/pytorch/issues/33303")
  x <- torch_rand(100, names = "a")
  y <- torch_zeros(1)
  expect_tensor(torch_var_out(y, x, dim = 0))
  expect_tensor(torch_var_out(y, x, dim = "a"))
})

test_that("where", {
  expect_equal_to_tensor(
    torch_where(torch_tensor(TRUE), torch_tensor(1), torch_tensor(0)),
    torch_tensor(1)
  )
  expect_equal_to_tensor(
    torch_where(torch_tensor(FALSE), torch_tensor(1), torch_tensor(0)),
    torch_tensor(0)
  )
  expect_tensor(
    torch_where(torch_tensor(c(TRUE, FALSE)), torch_ones(2), torch_zeros(2))
  )
})

test_that("zero_", {
  x <- torch_ones(2)
  y <- torch_zero_(x)
  expect_tensor(y)
  expect_equal_to_tensor(x, torch_tensor(c(0, 0)))
})

test_that("zeros", {
  expect_tensor(torch_zeros(c(2)))
  expect_equal_to_tensor(torch_zeros(2), torch_tensor(c(0, 0)))
  torch_zeros(2, names = "hello")
})

test_that("zeros_like", {
  x <- torch_ones(c(2))
  expect_tensor(y <- torch_zeros_like(x))
  expect_equal_to_tensor(y, torch_tensor(c(0, 0)))
  expect_tensor(torch_zeros_like(x, dtype = torch_int()))
})

test_that("zeros_out", {
  x <- torch_ones(c(2))
  expect_tensor(torch_zeros_out(x, c(2)))
  expect_equal_to_tensor(x, torch_tensor(c(0, 0)))
})
