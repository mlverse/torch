test_that("flatten and unflatten", {
  x <- torch_randn(32, 5, 3, 2)
  fl <- nn_flatten()
  unfl <- nn_unflatten(2, c(5, 3, 2))

  expect_equal_to_tensor(unfl(fl(x)), x)
})
