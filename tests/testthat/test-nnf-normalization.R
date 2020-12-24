test_that("nnf_normalize", {
  
  x <- torch_tensor(c(1,1,0,0))
  
  expect_error(nnf_normalize(x))
  expect_equal_to_tensor(
    nnf_normalize(x, dim = 1), 
    torch_tensor(c(0.7071, 0.7071, 0, 0)), 
    tolerance = 1e-5
  )
  
  out <- torch_empty(4)
  nnf_normalize(x, dim = 1, out = out)
  expect_equal_to_tensor(
    out, 
    torch_tensor(c(0.7071, 0.7071, 0, 0)), 
    tolerance = 1e-5
  )
  
})
