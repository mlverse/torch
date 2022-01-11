context("nn-init")

test_that("nn_init_eye", {
  w <- torch_empty(5, 5, requires_grad = TRUE)
  nn_init_eye_(w)

  expect_true(w$requires_grad)
  expect_equal_to_r(w, diag(nrow = 5, ncol = 5))
})
