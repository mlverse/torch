context("generator")

test_that("can create and use simple generators", {
  x <- torch_generator()
  x$set_current_seed(12345678L)
  expect_equal(x$current_seed(), bit64::as.integer64(12345678))

  x$set_current_seed(bit64::as.integer64(123456789101112))
  expect_equal(x$current_seed(), bit64::as.integer64(123456789101112))
})

test_that("manual_seed works", {
  torch_manual_seed(1L)
  a <- torch_randn(1)
  torch_manual_seed(1L)
  b <- torch_randn(1)

  expect_equal_to_tensor(a, b)

  torch_manual_seed(1L)
  a <- nn_linear(2, 2)
  torch_manual_seed(1L)
  b <- nn_linear(2, 2)

  expect_equal_to_tensor(a$weight, b$weight)
  expect_equal_to_tensor(a$bias, b$bias)
})

test_that("current behavior is identical to pytorch", {
  torch_manual_seed(1)
  expect_equal_to_r(torch_randn(1), 0.661352157592773)

  withr::with_options(new = list("torch.old_seed_behavior" = TRUE), {
    torch_manual_seed(2019)
    expect_equal_to_r(torch_randn(1), -0.364226043224335)
  })
})

test_that("can use torch bernoulli on cuda.", {
  # see https://github.com/mlverse/torch/issues/480
  expect_error({
    x <- torch_bernoulli(torch_ones(3,3, device="cpu") * 0.5)
  }, regexp = NA)
  expect_tensor_shape(x, c(3,3))
  
  skip_if_cuda_not_available()
  
  expect_error({
    x <- torch_bernoulli(torch_ones(3,3, device="cuda") * 0.5)
  }, regexp = NA)
  expect_tensor_shape(x, c(3,3))
  
}) 

test_that("torch manual seed works for CUDA", {
  torch_manual_seed(1)
  x <- torch_randn(1, device="cuda")
  
  torch_manual_seed(1)
  x_ <- torch_randn(1, device="cuda")
  
  expect_equal_to_tensor(x, x_)
})
  
test_that("Can use a with context to modify the torch seed temporarily", {
  
  torch_manual_seed(1)
  x <- torch_randn(1)
  y <- torch_randn(1)
  
  with_torch_manual_seed(seed = 1, {
    x_ <- torch_randn(1)
    y_ <- torch_randn(1)
    z_ <- torch_randn(1)
  })
  
  z <- torch_randn(1)
  
  expect_equal_to_tensor(x, x_)
  expect_equal_to_tensor(y, y_)
  expect_equal_to_tensor(z, z_)
  
})

test_that("The above also works for CUDA seeds", {
  skip_if_cuda_not_available()
  torch_manual_seed(1)
  x <- torch_randn(1, device="cuda")
  y <- torch_randn(1)
  
  with_torch_manual_seed(seed = 1, {
    x_ <- torch_randn(1, device="cuda")
    y_ <- torch_randn(1)
    w_ <- torch_randn(1, device="cuda")
    z_ <- torch_randn(1)
  })
  
  w <- torch_randn(1, device="cuda")
  z <- torch_randn(1)
  
  expect_equal_to_tensor(x, x_)
  expect_equal_to_tensor(y, y_)
  expect_equal_to_tensor(z, z_)
  expect_equal_to_tensor(w, w_)
})