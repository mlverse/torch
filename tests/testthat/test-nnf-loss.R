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
})
