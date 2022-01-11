context("nn-loss")

test_that("nn_bce_loss", {
  loss <- nn_bce_loss()
  input <- torch_tensor(c(0.1), requires_grad = TRUE)
  target <- torch_tensor(c(0))
  output <- loss(input, target)
  output$backward()

  expect_equal(as_array(output), -log(1 - 0.1) * 1, tolerance = 1e-7)
})

test_that("nn_cross_entropy_loss", {
  loss <- nn_cross_entropy_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randint(low = 1, high = 6, size = 3, dtype = torch_long())
  output <- loss(input, target)
  output$backward()

  expect_tensor(output)

  input <- torch_randn(1, 6, 100, 100, requires_grad = TRUE)
  target <- torch_randint(low = 1, high = 6, size = c(1, 100, 100), dtype = torch_long())
  expect_tensor(loss(input, target))
})

test_that("nn_kl_div_loss", {
  loss <- nn_kl_div_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randn(3, 5, requires_grad = TRUE)

  expect_warning(
    output <- loss(input, target)
  )

  output$backward()

  expect_tensor(output)
})

test_that("nn_hinge_embedding_loss", {
  loss <- nn_hinge_embedding_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randn(3, 5, requires_grad = TRUE)

  out <- loss(input, target)
  out$backward()

  expect_length(out$shape, 0)
})

test_that("multilabel margin loss", {
  loss <- nn_multilabel_margin_loss()
  x <- torch_tensor(c(0.1, 0.2, 0.4, 0.8))$view(c(1, 4))
  # for target y, only consider labels 4 and 1, not after label -1
  y <- torch_tensor(c(4, 1, -1, 2), dtype = torch_long())$view(c(1, 4))
  o <- loss(x, y)
  expect_equal(as.numeric(o), 0.85, tol = 1e-5)

  expect_length(o$shape, 0)
  y <- torch_tensor(c(4, 0, -1, 2), dtype = torch_long())$view(c(1, 4))
  expect_error(o <- loss(x, y))
})

test_that("smooth_l1_loss", {
  loss <- nn_smooth_l1_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randn(3, 5)
  o <- loss(input, target)

  expect_length(o$shape, 0)
})

test_that("soft_margin loss", {
  loss <- nn_soft_margin_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randn(3, 5)
  o <- loss(input, target)

  expect_length(o$shape, 0)
})

test_that("multilabel_soft_margin loss", {
  loss <- nn_multilabel_soft_margin_loss()
  input <- torch_randn(3, 5, requires_grad = TRUE)
  target <- torch_randn(3, 5)
  o <- loss(input, target)

  expect_length(o$shape, 0)
})

test_that("cosine_embedding loss", {
  loss <- nn_cosine_embedding_loss()
  input1 <- torch_randn(5, 5, requires_grad = TRUE)
  input2 <- torch_randn(5, 5, requires_grad = TRUE)
  target <- torch_randn(5)
  o <- loss(input1, input2, target)

  expect_length(o$shape, 0)
})

test_that("nn_multi_margin_loss", {
  loss <- nn_multi_margin_loss()
  input <- torch_randn(100, 5, requires_grad = TRUE)
  target <- torch_randint(low = 1, high = 5, size = c(100), dtype = torch_long())
  o <- loss(input, target)

  expect_length(o$shape, 0)
})
