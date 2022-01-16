#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py
#' TODO: add more unit tests

test_that("Bernoulli distribution - basic tests", {
  p <- torch_tensor(c(0.7, 0.2, 0.4), requires_grad = TRUE)
  r <- torch_tensor(0.3, requires_grad = TRUE)
  s <- 0.3

  expect_equal(distr_bernoulli(p)$sample(8)$size(), c(8, 3))
  expect_false(distr_bernoulli(p)$sample()$requires_grad)
  expect_equal(distr_bernoulli(r)$sample(8)$size(), c(8, 1))
  expect_equal(distr_bernoulli(r)$sample()$size(), 1)
  expect_equal(distr_bernoulli(s)$sample()$size(), 1)

  ref_log_prob <- function(idx, val, log_prob) {
    prob <- p[idx]
    prob <- if (as.logical(val != 0)) prob else 1 - prob
    expect_equal(log_prob, log(prob))
  }

  check_log_prob(distr_bernoulli(p), ref_log_prob)
  check_log_prob(distr_bernoulli(logits = p$log() - (-p)$log1p()), ref_log_prob)

  expect_error(distr_bernoulli(r)$rsample())

  # check entropy computation
  expect_equal(
    distr_bernoulli(p)$entropy(),
    torch_tensor(c(0.6108, 0.5004, 0.6730))
  )
  expect_equal(distr_bernoulli(0)$entropy(), torch_tensor(0))
  expect_equal(distr_bernoulli(s)$entropy(), torch_tensor(0.6108))
})

test_that("Bernoulli Distribution - enumerate support", {
  examples <- list(
    list(list(probs = 0.1), matrix(c(0, 1), 2, 1)),
    list(list(probs = c(0.1, 0.9)), matrix(c(0, 1), 2, 1)),
    list(
      list(probs = matrix(c(0.1, 0.1, 0.3, 0.4), 2, 2)),
      array(c(0, 1), dim = c(1, 2, 1))
    )
  )

  check_enumerate_support(distr_bernoulli, examples)
})

test_that("Bernoulli Distribution 3D", {
  p <- torch_full(c(2, 3, 5), 0.5)$requires_grad_()
  expect_equal(distr_bernoulli(p)$sample()$size(), c(2, 3, 5))
  expect_equal(
    distr_bernoulli(p)$sample(sample_shape = c(2, 5))$size(),
    c(2, 5, 2, 3, 5)
  )
  expect_equal(
    distr_bernoulli(p)$sample(2)$size(),
    c(2, 2, 3, 5)
  )
})

test_that("Bernoulli distribution - expand", {
  shapes <-
    list(NULL, 2, c(2, 1))

  d <- distr_bernoulli(torch_tensor(c(0.7, 0.2, 0.4),
    requires_grad = TRUE
  ))

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

test_that("Bernoulli distribution - enumerate_support", {
  d <- distr_bernoulli(0.7)
  required_values <- c(0, 1)
  unique_values <- unique(as.array(d$enumerate_support()))
  expect_true(all(unique_values %in% required_values))
})

test_that("log prob is correct", {
  probs <- torch_rand(10)
  d <- distr_bernoulli(probs = probs)

  x <- torch_tensor(sample(c(0, 1), 10, replace = TRUE))
  result <- d$log_prob(x)
  expected <- dbinom(as.numeric(x), 1, prob = as.numeric(probs), log = TRUE)

  expect_equal_to_r(result, expected, tol = 1e-6)
})

test_that("gradients are correct", {
  probs <- torch_tensor(c(0.5, 0.2), requires_grad = TRUE)
  d <- distr_bernoulli(probs = probs)

  x <- torch_cat(list(torch_ones(5, 2), torch_zeros(5, 2)))
  loss <- d$log_prob(x)$mean()
  loss$backward()

  expect_equal_to_r(probs$grad, c(0.0000000000, 0.9375000000)) # from pytorch
})
