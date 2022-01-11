#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py
#' TODO: add more unit tests

test_that("Gamma distribution - rsample", {
  num_samples <- 100

  for (alpha in c(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)) {
    alphas <- torch_tensor(
      rep(alpha, num_samples),
      dtype = torch_float(), requires_grad = TRUE
    )
    betas <- torch_tensor(rep(1, num_samples))

    x <- distr_gamma(alphas, betas)$rsample()
    x$sum()$backward()
    results <- x$sort()
    x <- results[[1]]
    ind <- results[[2]]
    ind <- as.array(ind)
    x <- as.array(x$detach())

    actual_grad <- as.array(alphas$grad[ind])

    eps <- 0.01 * alpha / (1.0 + alpha**0.5)
    cdf_alpha <- (pgamma(x, alpha + eps) - pgamma(x, alpha - eps)) / (2 * eps)
    cdf_x <- dgamma(x, alpha)
    expected_grad <- -cdf_alpha / cdf_x
    rel_error <- abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
    expect_lt(max(rel_error), 0.005)
  }
})


test_that("Gamma distribution - shape scalar params", {
  gamma <- distr_gamma(1, 1)
  scalar_sample <- 1
  tensor_sample_1 <- torch_ones(c(3, 2))
  tensor_sample_2 <- torch_ones(c(3, 2, 3))

  expect_equal(gamma$batch_shape, 1)
  expect_equal(gamma$event_shape, NULL)
  expect_equal(gamma$sample()$size(), 1)
  expect_equal(gamma$sample(c(3, 2))$size(), c(3, 2, 1))
  expect_equal(gamma$log_prob(scalar_sample)$size(), 1)
  expect_equal(gamma$log_prob(tensor_sample_1)$size(), c(3, 2))
  expect_equal(gamma$log_prob(tensor_sample_2)$size(), c(3, 2, 3))
})
