context("nn-loss")

test_that("nn_bce_loss", {
  loss <- nn_bce_loss()
  input <- torch_tensor(c(0.1), requires_grad=TRUE)
  target <- torch_tensor(c(0))
  output <- loss(input, target)
  output$backward()
  
  expect_equal(as_array(output), -log(1-0.1)*1, tolerance = 1e-7)
})

test_that("nn_cross_entropy_loss", {
  loss <- nn_cross_entropy_loss()
  input <- torch_randn(3, 5, requires_grad=TRUE)
  target <- torch_randint(low = 1, high = 6, size = 3, dtype = torch_long())
  output <- loss(input, target)
  output$backward()
  
  expect_tensor(output)
})
