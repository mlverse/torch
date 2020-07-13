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
