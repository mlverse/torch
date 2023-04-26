test_that("can access operators via ops object", {
  # matmul, default use
  res <- jit_ops$aten$matmul(torch::torch_ones(5, 4), torch::torch_rand(4, 5))
  expect_equal(length(res), 1)
  expect_equal(dim(res[[1]]), c(5, 5))
  
  # matmul, passing out tensor
  t1 <- torch::torch_ones(4, 4)
  t2 <- torch::torch_eye(4)
  out <- torch::torch_zeros(4, 4)
  jit_ops$aten$matmul(t1, t2, out)
  expect_equal_to_tensor(t1, out)
})

test_that("can print ops objects at different levels", {
  local_edition(3)
  expect_snapshot(jit_ops)
  expect_snapshot(jit_ops$sparse)
  expect_snapshot(jit_ops$prim$ChunkSizes)
  expect_snapshot(jit_ops$aten$fft_fft)
})


