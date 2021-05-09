test_that("multivariate nromal", {
  
  m <- distr_multivariate_normal(
    loc = torch_randn(2), 
    covariance_matrix = torch_eye(2)
  )
  
  expect_tensor_shape(m$sample(10), c(10, 2))
  
  x <- m$sample(10)
  expected_log_prob <- mvtnorm::dmvnorm(
    as.array(x),
    mean = as.array(m$loc), 
    sigma = as.array(m$covariance_matrix),
    log = TRUE
    )
  
  expect_equal_to_r(m$log_prob(x), expected_log_prob, tolerance = 1e-6)
  
})
