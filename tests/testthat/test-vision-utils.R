context("vision-utils")

test_that("vision_make_grid", {
  
  images <- torch_randn(c(4, 3, 16, 16))
  
  grid <- vision_make_grid(images, num_rows = 2, padding = 0)

  expect_equal(grid$size(), c(3, 32, 32))
  expect_equal(as.numeric(grid$max() - grid$min()), 1, tolerance = 1e-4)
  
})
