test_that("Can create a tensor", {
  x <- torch_tensor(1)
  
  expect_s3_class(x, "torch_tensor")
  
  x <- torch_tensor(1, dtype = torch_double())
  
  expect_s3_class(x, "torch_tensor")
})
