test_that("nnf_mse_loss", {

  x <- torch_tensor(c(1,2,3))
  y <- torch_tensor(c(2,3,4))
  
  o <- nnf_mse_loss(x, y)
  
  expect_equal_to_r(o, 1)
  
  y <- y$unsqueeze(2)
  
  expect_warning(
    nnf_mse_loss(x, y),
    regexp = "target size"
  )
  
})
