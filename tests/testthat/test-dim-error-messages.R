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

test_that("dim1 & dim2", {
  
  x <- torch_randn(2,2)
  
  expect_error(
    torch_transpose(x, 3, 1),
    regex = "Dimension out of range (expected to be in range of [-2, 2], but got 3)",
    fixed = TRUE
  )
  
  expect_error(
    torch_transpose(x, 2, 3),
    regex = "Dimension out of range (expected to be in range of [-2, 2], but got 3)",
    fixed = TRUE
  )
  
  expect_error(
    torch_diag_embed(x, dim1 = 4, dim2 = 1),
    "Dimension out of range (expected to be in range of [-3, 3], but got 4)",
    fixed = TRUE
  )
  
  expect_error(
    torch_diag_embed(x, dim1 = 3, dim2 = 4),
    "Dimension out of range (expected to be in range of [-3, 3], but got 4)",
    fixed= TRUE
  )
  
})

test_that("dimension x does not have size y", {
  
  a <- torch_randn(c(4, 3))
  b <- torch_randn(c(4, 3))
  
  expect_error(
    torch_cross(a, b, dim=1),
    regex = "dimension 1 does not have size 3",
    fixed = TRUE
  )
  
})

test_that("indices error message", {
  
  x <- torch_randn(4, 4, 2)
  e <- torch_max_pool2d_with_indices(x, kernel_size = c(2,2))
  
  i <- as_array(torch_max(e[[2]])$to(torch_int()))
  indices <- torch_ones_like(e[[2]])
  indices[1,..] <- 35L
  
  expect_error(
    torch_max_unpool2d(e[[1]], indices = indices, output_size = c(4,4)),
    paste0("Found an invalid max index: ", 35),
    fixed = TRUE
  )
  
})

test_that("torch_flatten dim error", {
  
  x <- torch_randn(2,2)
  expect_error(
    torch_flatten(x, start_dim = 3),
    "Dimension out of range (expected to be in range of [-2, 2], but got 3)",
    fixed = TRUE
  )
  
})