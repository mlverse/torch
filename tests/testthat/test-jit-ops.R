test_that("can access operators via ops object", {
  # matmul, default use
  res <- jit_ops$aten$matmul(torch::torch_ones(5, 4), torch::torch_rand(4, 5))
  expect_equal(dim(res), c(5, 5))
  
  # matmul, passing out tensor
  t1 <- torch::torch_ones(4, 4)
  t2 <- torch::torch_eye(4)
  out <- torch::torch_zeros(4, 4)
  jit_ops$aten$matmul(t1, t2, out)
  expect_equal_to_tensor(t1, out)
  
  # split, returning two tensors in a list of length 2
  res_torch <- torch_split(torch::torch_arange(0, 3), 2, 1)
  res_jit <- jit_ops$aten$split(torch::torch_arange(0, 3), jit_scalar(2L), jit_scalar(0L))
  expect_length(res_jit, 2)
  expect_equal_to_tensor(res_jit[[1]], res_torch[[1]])
  expect_equal_to_tensor(res_jit[[2]], res_torch[[2]])
  
  # split, returning a single tensor
  res_torch <- torch_split(torch::torch_arange(0, 3), 4, 1)
  res_jit <- jit_ops$aten$split(torch::torch_arange(0, 3), jit_scalar(4L), jit_scalar(0L))
  expect_length(res_jit, 1)
  expect_equal_to_tensor(res_jit[[1]], res_torch[[1]])
  
  # linalg_qr always returns a list
  m <- torch_eye(5)/5
  res_torch <- linalg_qr(m)
  res_jit <- jit_ops$aten$linalg_qr(m, jit_scalar("reduced"))
  expect_equal_to_tensor(res_torch[[2]], res_jit[[2]])
})

test_that("can print ops objects at different levels", {
  local_edition(3)
  expect_snapshot(jit_ops)
  expect_snapshot(jit_ops$sparse)
  expect_snapshot(jit_ops$prim$ChunkSizes)
  expect_snapshot(jit_ops$aten$fft_fft)
})






