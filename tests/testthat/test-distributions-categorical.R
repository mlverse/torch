test_that("categorical distribution works", {
  p <- torch_tensor(c(0.1, 0.2, 0.3), requires_grad = TRUE)
  d <- distr_categorical(probs = p)

  expect_equal(d$sample()$shape, integer(0))
  expect_equal(d$sample()$requires_grad, FALSE)
  expect_equal(d$sample(c(2, 2))$shape, c(2, 2))
  expect_equal(d$sample(c(1))$shape, c(1))
  expect_equal_to_r(torch_all(torch_isnan(d$mean)), TRUE)
  expect_equal_to_r(torch_all(torch_isnan(d$variance)), TRUE)

  expect_equal_to_tensor(
    d$log_prob(torch_tensor(c(1, 2, 3))),
    d$logits
  )
})

test_that("categorical 2d works", {
  prob1 <- rbind(c(0.1, 0.2, 0.3), c(0.5, 0.3, 0.2))
  prob2 <- rbind(c(1.0, 0.0), c(0.0, 1.0))

  p1 <- torch_tensor(prob1, requires_grad = TRUE)
  p2 <- torch_tensor(prob2, requires_grad = TRUE)

  expect_equal(distr_categorical(probs = p1)$sample()$shape, 2)
  expect_equal(distr_categorical(probs = p2)$sample()$shape, 2)
})
