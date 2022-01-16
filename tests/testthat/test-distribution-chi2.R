#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py

test_that("Chi2 basic test", {
  num_samples <- 100
  for (df in c(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)) {
    dfs <- torch_tensor(
      rep(df, num_samples),
      dtype = torch_float(), requires_grad = TRUE
    )
    x <- distr_chi2(dfs)$rsample()
    x$sum()$backward()
    out <- x$sort()
    x <- out[[1]]
    ind <- out[[2]]
    x <- as.array(x$detach())
    actual_grad <- dfs$grad[ind]

    # Compare with expected gradient dx/ddf along constant cdf(x,df).
    eps <- 0.01 * df / (1.0 + df**0.5)
    cdf_df <- (pchisq(x, df + eps) - pchisq(x, df - eps)) / (2 * eps)
    cdf_x <- dchisq(x, df)
    expected_grad <- -cdf_df / cdf_x
    rel_error <- abs(actual_grad - expected_grad) / (expected_grad + 1e-30)

    expect_lt(as.array(max(rel_error)), 0.005)
  }
})

test_that("Chi2 shape", {
  df <- torch_randn(2, 3)$exp()$requires_grad_()
  df_1d <- torch_randn(1)$exp()$requires_grad_()

  expect_equal(distr_chi2(df)$sample()$size(), c(2, 3))
  expect_equal(distr_chi2(df)$sample(5)$size(), c(5, 2, 3))
  expect_equal(distr_chi2(df_1d)$sample(1)$size(), c(1, 1))
  expect_equal(distr_chi2(df_1d)$sample()$size(), c(1))
  expect_equal(distr_chi2(torch_tensor(0.5, requires_grad = TRUE))$sample()$size(), 1)
  expect_equal(distr_chi2(0.5)$sample()$size(), 1)
  expect_equal(distr_chi2(0.5)$sample(1)$size(), c(1, 1))

  ref_log_prob <- function(idx, x, log_prob) {
    d <- df$view(-1)[idx]$detach()
    expected <- log(dchisq(as.array(x), as.array(d)))
    expect_equal(as.array(log_prob), expected, tolerance = 1e-2)
  }

  check_log_prob(distr_chi2(df), ref_log_prob)
})

test_that("Chi2 shape tensor params", {
  chi2 <- distr_chi2(torch_tensor(c(1., 1.)))

  tensor_sample_1 <- torch_ones(3, 2)
  tensor_sample_2 <- torch_ones(3, 2, 3)

  expect_equal(chi2$batch_shape, 2)
  expect_equal(chi2$event_shape, NULL)
  expect_equal(chi2$sample()$size(), 2)
  expect_equal(chi2$sample(c(3, 2))$size(), c(3, 2, 2))
  expect_equal(chi2$log_prob(tensor_sample_1)$size(), c(3, 2))
  expect_error(chi2$log_prob(tensor_sample_2))
  expect_equal(chi2$log_prob(torch_ones(2, 1))$size(), c(2, 2))
})

test_that("Chi2 shape scalar params", {
  chi2 <- distr_chi2(1)

  tensor_sample_1 <- torch_ones(3, 2)
  tensor_sample_2 <- torch_ones(3, 2, 3)

  expect_equal(chi2$batch_shape, 1)
  expect_equal(chi2$event_shape, NULL)
  expect_equal(chi2$sample()$size(), 1)
  expect_equal(chi2$sample(c(3, 2))$size(), c(3, 2, 1))
  expect_equal(chi2$log_prob(tensor_sample_1)$size(), c(3, 2))
  expect_equal(chi2$log_prob(tensor_sample_2)$size(), c(3, 2, 3))
})
