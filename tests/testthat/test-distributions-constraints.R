test_that("constraint real vector", {
  
  x <- torch_randn(5)
  expect_equal_to_r(constraint_real_vector$check(x), TRUE)
  
  x <- torch_randn(5, 5)
  expect_equal_to_r(constraint_real_vector$check(x), rep(TRUE, 5))
  
  x <- log(torch_randn(5, 5) -3)
  expect_equal_to_r(constraint_real_vector$check(x), rep(FALSE, 5))
  
})
