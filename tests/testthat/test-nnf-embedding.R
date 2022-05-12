test_that("nnf_one_hot", {
  expect_tensor_shape(nnf_one_hot(torch_tensor(1L)), c(1, 1))
  expect_tensor_shape(nnf_one_hot(torch_tensor(c(1L, 2L))), c(2, 2))
  expect_error(nnf_one_hot(torch_tensor(0L)))
})

test_that("nnf_embedding_bag", {
  input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
  weight <- torch_randn(10, 3)
  
  out <- nnf_embedding_bag(input, weight)
  
  expect_equal(out[1,], (weight[1,] + weight[2,] + weight[4,] + weight[5,])/4)
  expect_equal(out[2,], (weight[4,] + weight[3,] + weight[2,] + weight[9,])/4)
  expect_error(nnf_embedding_bag(input, weight, offsets=torch_tensor(c(1,2))))
})