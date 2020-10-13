test_that("result_type", {
  
  x <- torch_result_type(
    tensor1 = torch_tensor(c(1, 2), dtype=torch_int()), 
    tensor2 = 1
  )
  
  expect_true(x == torch_float())
  
  x <- torch_result_type(
    tensor1 = torch_tensor(c(1, 2), dtype=torch_int()), 
    tensor2 = torch_tensor(1:2)
  )
  
  expect_true(x == torch_long())
  
  x <- torch_result_type(
    tensor1 = 1, 
    tensor2 = torch_tensor(1:2)
  )
  
  expect_true(x == torch_float())
  
  x <- torch_result_type(
    tensor1 = 1, 
    tensor2 = 2L
  )
  
  expect_true(x == torch_float())
  
})

test_that("torch_multi_margin_loss", {
  
  x <- torch_randn(3, 2)
  y <- torch_tensor(c(1, 2, 3), dtype = torch_long())
  
  expect_error(torch_multi_margin_loss(x, y))
  
  x <- torch_randn(3, 3)
  expect_tensor(torch_multi_margin_loss(x, y))
  
  y <- torch_tensor(c(0, 1, 2))
  expect_error(torch_multi_margin_loss(x, y))
  
})

test_that("torch_topk", {
  
  x <- torch_arange(1, 16)$view(c(5, 3))
  expect_equal_to_r(torch_topk(x, 2)[[2]], 
                    matrix(c(3, 2), nrow = 5, ncol = 2, byrow = TRUE))
  
  expect_equal_to_r(x$topk(2)[[2]], 
                    matrix(c(3, 2), nrow = 5, ncol = 2, byrow = TRUE))
  
})

test_that("torch_narrow", {
  
  x <- torch_tensor(matrix(1:9, ncol = 3, byrow = TRUE))
  expect_equal_to_tensor(torch_narrow(x, 1, 1, 2), x[1:2,])
  expect_equal_to_tensor(x$narrow(1, 1, 2), x[1:2,])
  expect_equal_to_tensor(x$narrow_copy(1, 1, 2), x[1:2,])
  
})
