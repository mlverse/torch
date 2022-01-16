test_that("multivariate nromal", {
  skip_if_not_installed(c("mvtnorm"))

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

test_that("multivaraitae_normal additional shapes", {
  m <- distr_multivariate_normal(
    loc = torch_randn(2),
    covariance_matrix = torch_eye(2)
  )

  expect_tensor_shape(m$sample(c(10, 10)), c(10, 10, 2))
  expect_tensor_shape(m$log_prob(m$sample(c(10, 10))), c(10, 10))

  expect_tensor_shape(m$sample(c(10, 5, 3)), c(10, 5, 3, 2))
  expect_tensor_shape(m$log_prob(m$sample(c(10, 5, 3))), c(10, 5, 3))
})

test_that("multivariate normal gradients", {
  skip_if_not_installed(c("numDeriv", "mvtnorm"))

  loc <- torch_randn(2, requires_grad = TRUE)
  var <- torch_eye(2, requires_grad = TRUE)

  m <- distr_multivariate_normal(
    loc = loc,
    covariance_matrix = var
  )

  sample <- m$sample(10)
  loss <- m$log_prob(sample)$mean()
  loss$backward()

  grad_mean <- numDeriv::grad(
    func = function(x) {
      mean(mvtnorm::dmvnorm(
        as.array(sample),
        mean = x,
        sigma = as.array(m$covariance_matrix),
        log = TRUE
      ))
    },
    x = as.array(m$loc)
  )

  grad_sigma <- numDeriv::grad(
    func = function(x) {
      mean(mvtnorm::dmvnorm(
        as.array(sample),
        mean = as.array(m$loc),
        sigma = x,
        log = TRUE,
        checkSymmetry = FALSE
      ))
    },
    x = as.array(m$covariance_matrix)
  )

  expect_equal_to_r(loc$grad, grad_mean, tolerance = 1e-6)
  expect_equal_to_r(var$grad$view(4)[c(1, 4)], grad_sigma[c(1, 4)], tolerance = 1e-6)
})

test_that("properites", {
  m <- distr_multivariate_normal(
    loc = torch_randn(2),
    covariance_matrix = torch_eye(2)
  )

  expect_equal_to_tensor(m$scale_tril, torch_eye(2))
  expect_equal_to_tensor(m$covariance_matrix, torch_eye(2))
  expect_equal_to_tensor(m$precision_matrix, torch_eye(2))
  expect_equal_to_tensor(m$mean, m$loc)
  expect_tensor_shape(m$variance, c(2))
  expect_tensor_shape(m$entropy(), 1)
})
