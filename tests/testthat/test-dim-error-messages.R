test_that("out of bound error message", {
  
  x <- torch_tensor(1)
  
  funs <- list(
    torch_sum,
    torch_mean,
    torch_median
  )
  
  for (f in funs) {
    expect_error(
      f(x, dim = 2),
      regex = "Dimension out of range (expected to be in range of [-1, 1], but got 2)",
      fixed = TRUE
    )  
  }
  
  x <- torch_randn(size = rep(1, 10))
  
  for (f in funs) {
    expect_error(
      f(x, dim = 11),
      regex = "Dimension out of range (expected to be in range of [-10, 10], but got 11)",
      fixed = TRUE
    )  
  }
  
})
