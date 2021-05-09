test_that("multivariate nromal", {
  
  m <- distr_multivariate_normal(
    loc = torch_randn(2), 
    covariance_matrix = torch_eye(2)
  )
  
})