test_that("can run convert to and back", {
  
  x <- list(torch_tensor(1))
  expect_equal(test_stack(x), x)
  
  x <- list(torch_tensor(1), 1)
  expect_equal(test_stack(x), x)
  
  x <- list(torch_tensor(1), 1, c(1,2,3))
  expect_equal(test_stack(x), x)
  
  x <- list(torch_tensor(1), 1, c(1,2,3), 1L, 1:10)
  expect_equal(test_stack(x), x)
  
  x <- list(torch_tensor(1), 1, c(1,2,3), 1L, 1:10, list(torch_tensor(1), torch_tensor(2)))
  expect_equal(test_stack(x), x)
  
  x <- list(torch_tensor(1), 1, c(1,2,3), list(1L, 1:10), list(torch_tensor(1), torch_tensor(2)))
  expect_equal(test_stack(x), x)
  
})
