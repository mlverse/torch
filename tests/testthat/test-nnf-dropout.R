test_that("nnf_dropout2d raise warnings", {
  x <- torch_randn(3, 10, 10)
  expect_warning(nnf_dropout2d(x))
  
  x <- torch_randn(3, 10)
  expect_warning(nnf_dropout2d(x))
  
  x <- nnf_dropout2d(torch_randn(1,2,10,10), p = 1)
  expect_true(torch_allclose(x, torch_zeros_like(x)))
})
