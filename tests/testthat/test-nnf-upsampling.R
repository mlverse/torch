test_that("interpolate", {
  img <- torch_ones(3, 32, 32)

  expect_tensor_shape(nnf_interpolate(img, size = 40), c(3, 32, 40))

  img <- torch_ones(1, 3, 32, 32)
  o <- nnf_interpolate(img, size = c(40, 40), mode = "bilinear")
  expect_tensor_shape(o, c(1, 3, 40, 40))
  expect_true(!any(as_array(torch_isnan(o)))) # no nans
})
