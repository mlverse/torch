test_that("norm works", {
  
  a <- torch_arange(0, 8, dtype=torch_float()) - 4
  b <- a$reshape(c(3, 3))

  expect_equal_to_tensor(linalg_norm(a), linalg_norm(b))
  expect_equal_to_tensor(linalg_norm(a), linalg_norm(b, ord = "fro"))
  expect_equal_to_r(linalg_norm(a, Inf), 4)
  expect_equal_to_r(linalg_norm(a, -Inf), 0)
  expect_equal_to_r(linalg_norm(b, Inf), 9)
  expect_equal_to_r(linalg_norm(b, -Inf), 2)
  expect_equal_to_r(linalg_norm(a, 1), 20)
  expect_equal_to_r(linalg_norm(b, 1), 7)
  expect_equal_to_r(linalg_norm(a, -1), 0)
  expect_equal_to_r(linalg_norm(b, -1), 6)
  expect_equal_to_tensor(linalg_norm(a, 2), linalg_norm(b))
  expect_equal_to_r(linalg_norm(a, -2), 0)
  expect_equal_to_r(linalg_norm(a, -3), 0)
  expect_equal_to_r(linalg_norm(a, -2), 0)
  
  expect_equal(linalg_norm(b, dim = 2)$numel(), 3)
  expect_equal(linalg_norm(b, dim = 1)$numel(), 3)
  
  expect_true(linalg_norm(a, dtype = torch_double())$dtype == torch_double())
})

test_that("vector norm works", {
  
  a <- torch_arange(0, 8, dtype=torch_float()) - 4
  b <- a$reshape(c(3, 3))
  
  expect_equal_to_tensor(linalg_vector_norm(a), linalg_vector_norm(b))
  
})

test_that("matrix norm", {
  
  a <- torch_arange(0, 8, dtype=torch_float())$reshape(c(3,3))
  expect_equal_to_r(linalg_matrix_norm(a, ord = -1), 9)
  
  b <- a$expand(c(2, -1, -1))
  expect_equal(linalg_matrix_norm(b)$numel(), 2)
  expect_equal(linalg_matrix_norm(b, dim = c(1, 3))$numel(), 3)
  
})

test_that("det works", {
  a <- torch_randn(3,3)
  expect_tensor_shape(linalg_det(a), integer(0))
  
  a <- torch_randn(3,3,3)
  expect_tensor_shape(linalg_det(a), c(3))
  
})
