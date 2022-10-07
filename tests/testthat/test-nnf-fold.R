test_that("unfold works", {
  input <- torch_randn(2, 5, 3, 4)
  output <- nnf_unfold(input, kernel_size = c(2,3))
  expect_equal(output$size(), c(2, 30, 4))
  
  input <- torch_randn(2, 5, 3, 4, 5)
  expect_error(
    output <- nnf_unfold(input, kernel_size = c(2,3)),
    regexp = "Expected 3D or 4D"
  )
})
