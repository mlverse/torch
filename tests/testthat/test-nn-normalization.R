test_that("layer_norm", {

  input <- torch_tensor(t(matrix(1:3, ncol = 3, nrow = 3)), dtype = torch_float())
  
  m <- nn_layer_norm(3, elementwise_affine = TRUE)
  result <- matrix(
    c(-1.22473537921906, -1.22473537921906, -1.22473537921906, 
      0, 0, 0, 1.22473549842834, 1.22473549842834, 1.22473549842834), 
    nrow = 3, ncol = 3
  )
  expect_equal_to_r(
    m(input),
    result,
    tolerance = 1e-6
  )
  
  m <- nn_layer_norm(3, elementwise_affine = FALSE)
  expect_equal_to_r(
    m(input),
    result,
    tolerance = 1e-6
  )
  
  input <- torch_randn(3,4,5)
  m <- nn_layer_norm(input$size()[-1])
  expect_tensor_shape(m(input), c(3,4,5))
  
})
