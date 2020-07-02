context("tensor-list")

test_that("torch_tensor_list works correctly", {
  
  x <- list(torch_tensor(1), torch_tensor(2))
  obj <- torch_tensor_list(x)
  y <- obj$to_r()
  
  expect_length(y, 2)
  expect_s3_class(y[[1]], "torch_tensor")
})
