test_that("nnf_mse_loss", {
  x <- torch_tensor(c(1, 2, 3))
  y <- torch_tensor(c(2, 3, 4))

  o <- nnf_mse_loss(x, y)

  expect_equal_to_r(o, 1)

  y <- y$unsqueeze(2)

  expect_warning(
    nnf_mse_loss(x, y),
    regexp = "target size"
  )
})

test_that("nnf_binary_cross_entropy", {
  x <- torch_tensor(c(0, 1))$view(c(-1, 1))
  y <- torch_tensor(c(0, 1))$view(c(-1, 1))

  expect_equal_to_r(nnf_binary_cross_entropy(x, y), 0)
})

test_that("nnf_nll_loss", {

  # test branch entered for dim == 3
  x <- torch_randn(32, 10, 5)
  y <- torch_randint(1, 10, size = list(32, 5), dtype = torch_long())

  o <- nnf_nll_loss(x, y)
  expect_length(o$size(), 0)
  
  # test ignore index
  x <- torch_randn(2, 10)
  y <- torch_tensor(c(1, -100), dtype = torch_int64())
  
  o <- nnf_nll_loss(x, y)
  o2 <- nnf_nll_loss(x[1,], y[1])
  expect_equal_to_tensor(o, o2)
})

test_that("l1 loss", {
  input <- torch_randn(3, 5, requires_grad=TRUE)
  target <- torch_randn(3, 5)
  output <- nnf_l1_loss(input, target)
  expect_equal(output$size(), integer(0))
  output$backward()
  expect_tensor(input$grad)
})

test_that("multilabel margin loss", {
  x <- torch_tensor(matrix(c(0.1, 0.2, 0.4, 0.8), nrow=1))
  y <- torch_tensor(matrix(c(4, 1, -1, 2), nrow = 1), dtype = torch_int64())
  out <- nnf_multilabel_margin_loss(x, y)
  expect_equal(out$item(), 0.85, tolerance = 1e-6)
})

test_that("multilabel soft margin loss", {
  x <- torch_tensor(matrix(c(0.1, 0.2, 0.4, 0.8), nrow=1))
  y <- torch_tensor(matrix(c(0, 1, 1, 0), nrow = 1), dtype = torch_int64())
  out <- nnf_multilabel_soft_margin_loss(x, y)
  expect_equal(out$item(), 0.7567, tolerance = 1e-4)
})
