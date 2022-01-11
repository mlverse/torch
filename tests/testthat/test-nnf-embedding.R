test_that("nnf_one_hot", {
  expect_tensor_shape(nnf_one_hot(torch_tensor(1L)), c(1, 1))
  expect_tensor_shape(nnf_one_hot(torch_tensor(c(1L, 2L))), c(2, 2))
  expect_error(nnf_one_hot(torch_tensor(0L)))
})
