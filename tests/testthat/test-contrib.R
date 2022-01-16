test_that("multiplication works", {
  skip_if_cuda_not_available()

  v <- torch_randn(8, 1024, 24, 2)$cuda()
  mean <- torch_mean(v, dim = 2, keepdim = TRUE)
  v <- v - mean
  m <- (torch_rand(8, 1024, 24) > 0.8)$cuda()
  nv <- torch_sum(m$to(dtype = torch_int()), dim = -1)$to(dtype = torch_int())$cuda()
  result <- contrib_sort_vertices(v, m, nv)

  expect_tensor_shape(result, c(8, 1024, 9))
})
