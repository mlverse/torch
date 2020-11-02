test_that("max with indices", {
  x <- torch_tensor(c(5,6,7,8))
  m <- torch_max(x, dim = 1)
  
  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 4)
})

test_that("min with indices", {
  x <- torch_tensor(c(5,6,7,8))
  m <- torch_min(x, dim = 1)
  
  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 1)
})

test_that("argsort", {
  
  x <- torch_tensor(c(3,2,1))
  expect_equal_to_r(torch_argsort(x), c(3, 2, 1))
  expect_equal_to_r(x$argsort(), c(3, 2, 1))
  
  x <- torch_tensor(c(1,2,3))
  expect_equal_to_r(torch_argsort(x, descending = TRUE), c(3, 2, 1))
  expect_equal_to_r(x$argsort(descending = TRUE), c(3, 2, 1))
  
  x <- torch_tensor(1:10)$view(c(5,2))
  expect_equal_to_r(torch_argsort(x, dim = 1)[,1], 1:5)
  expect_equal_to_r(x$argsort(dim = 1)[,1], 1:5)
  
  expect_equal_to_r(torch_argsort(x, dim = 2)[,1], rep(1, 5))
  expect_equal_to_r(x$argsort(dim = 2)[,1], rep(1, 5))
  
})
