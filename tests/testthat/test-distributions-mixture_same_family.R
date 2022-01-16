test_that("mixture same family works", {
  mix <- distr_categorical(torch_ones(5))
  comp <- distr_normal(torch_randn(5), torch_rand(5))
  gmm <- distr_mixture_same_family(mix, comp)

  expect_s3_class(gmm, "torch_Distribution")
  expect_s3_class(gmm, "torch_MixtureSameFamily")
  expect_tensor_shape(gmm$sample(), integer(0))
  expect_tensor_shape(gmm$sample(1), 1)
  expect_tensor_shape(gmm$sample(c(2, 2)), c(2, 2))
})

test_that("log prob  and cdf are equal to reference", {
  probs <- torch_tensor(c(0.6, 0.4))
  loc <- torch_zeros(2)
  scale <- torch_ones(2)

  d <- distr_mixture_same_family(
    distr_categorical(probs),
    distr_normal(loc = loc, scale = scale)
  )

  result <- d$log_prob(torch_tensor(rbind(c(1, 2), c(0, -1))))
  # reference from python
  expected <- torch_tensor(rbind(
    c(-1.4189383984, -2.9189386368),
    c(-0.9189383984, -1.4189383984)
  ))

  expect_equal_to_tensor(result, expected, tol = 1e-5)

  result <- d$cdf(torch_tensor(rbind(c(1, 2), c(0, -1))))
  expected <- torch_tensor(rbind(
    c(0.8413447738, 0.9772499204),
    c(0.5000000000, 0.1586552560)
  ))

  expect_equal_to_tensor(result, expected, tol = 1e-5)
})

test_that("gradients are similar to python", {
  probs <- torch_tensor(c(0.6, 0.4), requires_grad = TRUE)
  loc <- torch_zeros(2, requires_grad = TRUE)
  scale <- torch_ones(2, requires_grad = TRUE)

  d <- distr_mixture_same_family(
    distr_categorical(probs),
    distr_normal(loc = loc, scale = scale)
  )

  loss <- d$log_prob(torch_tensor(rbind(c(1, 2), c(0, -1))))$mean()
  loss$backward()

  expect_equal_to_r(probs$grad, c(-9.9341050941e-09, 1.4901161194e-08), tol = 1e-6)
  expect_equal_to_r(loc$grad, c(0.3000000119, 0.1999999881), tol = 1e-6)
  expect_equal_to_r(scale$grad, c(0.2999999523, 0.1999999881), tol = 1e-6)
})
