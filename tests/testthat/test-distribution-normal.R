#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py
#' TODO: add more unit tests

test_that("Distribution Normal - basic size test", {
  loc <- torch_randn(5, 5, requires_grad = TRUE)
  scale <- torch_randn(5, 5)$abs()$requires_grad_()
  loc_1d <- torch_randn(1, requires_grad = TRUE)
  scale_1d <- torch_randn(1)$abs()$requires_grad_()

  expect_equal(distr_normal(loc, scale)$sample()$size(), c(5, 5))
  expect_equal(distr_normal(loc, scale)$sample(7)$size(), c(7, 5, 5))
  expect_equal(distr_normal(loc_1d, scale_1d)$sample(1)$size(), c(1, 1))
  expect_equal(distr_normal(loc_1d, scale_1d)$sample()$size(), 1)
  expect_equal(distr_normal(0.2, .6)$sample(1)$size(), c(1, 1))
  expect_equal(distr_normal(-0.7, 50.0)$sample()$size(), 1)

  # Sample check for extreme value of mean, std
  loc_delta <- torch_tensor(c(1.0, 0.0))
  scale_delta <- torch_tensor(c(1e-5, 1e-5))
  expect_equal(
    distr_normal(loc_delta, scale_delta)$sample(sample_shape = c(1, 2)),
    torch_tensor(c(1.0, 0.0, 1.0, 0.0))$reshape(c(2, 2))
  )

  # Check gradient
  eps <- torch_normal(torch_zeros_like(loc), torch_ones_like(scale))
  z <- distr_normal(loc, scale)$rsample()
  z$backward(torch_ones_like(z))

  expect_equal(loc$grad, torch_ones_like(loc))
  expect_equal(scale$grad, eps)

  loc$grad$zero_()
  scale$grad$zero_()
  expect_equal(z$size(), c(5, 5))
})

test_that("Distribution Normal - expand", {
  shapes <-
    list(NULL, 2, c(2, 1))

  d <- distr_normal(loc = 1, scale = 1)

  for (shape in shapes) {
    shape <- shape[[1]]
    expanded_shape <- c(shape, d$batch_shape)
    original_shape <- c(d$batch_shape, d$event_shape)
    expected_shape <- c(shape, original_shape)
    expanded <- d$expand(batch_shape = c(expanded_shape))
    sample <- expanded$sample()
    actual_shape <- expanded$sample()$shape

    expect_equal(class(expanded), class(d))
    expect_equal(d$sample()$shape, original_shape)
    expect_equal(expanded$log_prob(sample), d$log_prob(sample))
    expect_equal(actual_shape, expected_shape)
    expect_equal(expanded$batch_shape, expanded_shape)
  }
})
