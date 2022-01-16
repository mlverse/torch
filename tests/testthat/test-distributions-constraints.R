test_that("constraint real vector", {
  x <- torch_randn(5)
  expect_equal_to_r(constraint_real_vector$check(x), TRUE)

  x <- torch_randn(5, 5)
  expect_equal_to_r(constraint_real_vector$check(x), rep(TRUE, 5))

  x <- log(torch_randn(5, 5) - 3)
  expect_equal_to_r(constraint_real_vector$check(x), rep(FALSE, 5))
})

test_that("positive definite", {
  x <- torch_randn(10, 2, 2)
  expect_tensor_shape(constraint_positive_definite$check(x), c(1, 10))

  x <- torch_ones(10, 2, 2)
  expect_equal_to_r(constraint_positive_definite$check(x), matrix(FALSE, nrow = 1, ncol = 10))

  x <- torch_eye(2, 2)$unsqueeze(1)
  expect_equal_to_r(constraint_positive_definite$check(x), matrix(TRUE, nrow = 1, ncol = 1))
})

test_that("lower cholesky", {
  x <- torch_rand(5, 2, 2)
  expect_equal_to_r(constraint_lower_cholesky$check(x), rep(FALSE, 5))

  x <- torch_randn(5, 2, 2)
  expect_equal_to_r(constraint_lower_cholesky$check(x), rep(FALSE, 5))

  x <- torch_eye(2, 2)$unsqueeze(1)
  expect_equal_to_r(constraint_lower_cholesky$check(x), rep(TRUE, 1))

  x <- torch_eye(2, 2)$unsqueeze(1)$mul(-2)
  expect_equal_to_r(constraint_lower_cholesky$check(x), rep(FALSE, 1))
})
