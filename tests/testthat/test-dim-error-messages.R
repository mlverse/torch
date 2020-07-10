test_that("out of bound error message", {
  
  x <- torch_tensor(1)
  
  funs <- list(
    torch_sum,
    torch_mean,
    torch_median,
    torch_max,
    torch_min,
    torch_cummax,
    torch_cummin,
    torch_mode,
    torch_size,
    torch_softmax,
    torch_std,
    torch_squeeze,
    torch_var
  )
  
  for (f in funs) {
    expect_error(
      f(x, dim = 2),
      regex = "Dimension out of range (expected to be in range of [-1, 1], but got 2)",
      fixed = TRUE
    )  
  }
  
  for (f in funs) {
    expect_error(
      f(x, dim = -2),
      regex = "Dimension out of range (expected to be in range of [-1, 1], but got -2)",
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

test_that("more than 1 dim", {
  
  x <- torch_randn(2,2,2)
 
  expect_error(
    torch_sum(x, dim = c(1,4)),
    regex = "Dimension out of range (expected to be in range of [-3, 3], but got 4)",
    fixed = TRUE
  )
  
})
