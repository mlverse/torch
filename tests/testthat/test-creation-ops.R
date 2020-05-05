test_that("torch_ones", {
  x <- torch_ones(5,5,
                  dtype = torch_float32(), layout = torch_strided(), 
                  device = torch_device("cpu"), requires_grad = TRUE)
  
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))
  
  x <- torch_ones(c(5,5))
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))
  
  x <- torch_ones(size = c(5,5))
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))
})

test_that("ones_like", {
  x <- torch_ones(2,2)
  y <- torch_ones_like(x)
  expect_equal_to_tensor(x, y)
})

test_that("rand", {
  x <- torch_rand(2,2,2)
  expect_equal(dim(as_array(x)), c(2,2,2))
})

test_that("rand_like", {
  x <- torch_rand(2,2,2)
  y <- torch_rand_like(x)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})

test_that("randint", {
  x <- torch_randint(0, 10, c(2,2))
  expect_equal(dim(as_array(x)), c(2,2))
  
  x <- torch_randint(0, 10, c(2,2), generator = torch_generator())
  expect_equal(dim(as_array(x)), c(2,2))
})

test_that("randint_like", {
  x <- torch_randint(0, 10, c(2,2))
  y <- torch_randint_like(x, 0, 500)
  expect_equal(dim(as_array(x)), dim(as_array(y)))
})