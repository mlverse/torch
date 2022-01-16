#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py
#' TODO: add more unit tests

test_that("Poisson distribution - shape test", {
  rate <- torch_randn(2, 3)$abs()$requires_grad_()
  rate_1d <- torch_randn(1)$abs()$requires_grad_()
  expect_equal(distr_poisson(rate)$sample()$size(), c(2, 3))
  expect_equal(distr_poisson(rate_1d)$sample()$size(), 1)
  expect_equal(distr_poisson(rate_1d)$sample(1)$size(), c(1, 1))
  expect_equal(distr_poisson(2.0)$sample(2)$size(), c(2, 1))
})

test_that("Poisson distribution - log_prob", {
  rate <- torch_randn(2, 3)$abs()$requires_grad_()
  rate_1d <- torch_randn(1)$abs()$requires_grad_()

  ref_log_prob <- function(idx, x, log_prob) {
    l <- rate$view(-1)[idx]$detach()
    l <- as.vector(as.array(l))
    x <- as.vector(as.array(x))
    expected <- ppois(x, l, log.p = TRUE)
    expect_equal(log_prob, torch_tensor(expected))
  }

  check_log_prob(distr_poisson(rate), ref_log_prob)
})
