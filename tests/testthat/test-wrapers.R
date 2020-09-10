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
