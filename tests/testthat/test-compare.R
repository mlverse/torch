skip_if(getRversion() <= "4.0.0")

test_that("comparison takes requires_grad into account", {
  testthat::local_edition(3)
  expect_equal(
    torch_tensor(1)$requires_grad_(FALSE),
    torch_tensor(1)$requires_grad_(FALSE)
  )
  expect_equal(
    torch_tensor(1)$requires_grad_(TRUE),
    torch_tensor(1)$requires_grad_(TRUE)
  )
  expect_failure(expect_equal(
    torch_tensor(1)$requires_grad_(FALSE),
    torch_tensor(1)$requires_grad_(TRUE)
  ))
})

test_that("comparison takes tensor's value into account", {
  testthat::local_edition(3)
  expect_failure(expect_equal(
    torch_tensor(1),
    torch_tensor(2)
  ))
})

test_that("comparison takes tensor's dimension into account", {
  testthat::local_edition(3)
  expect_failure(expect_equal(
    torch_tensor(1)$reshape(c(1, 1)),
    torch_tensor(1)$reshape(1)
  ))
})

test_that("grad_fn is respected", {
  testthat::local_edition(3)
  x = torch_tensor(1)$requires_grad_(TRUE)
  # grad_fn is changed after cloning
  expect_failure(expect_equal(
    x,
    x$clone()
  ))

  # without requires_grad, grad_fn is not changed

  x = torch_tensor(1)
  # grad_fn is changed
  expect_equal(
    x,
    x$clone()
  )
})

test_that("compare tensors using cuda", {
  skip_if_cuda_not_available()
  testthat::local_edition(3)

  expect_failure(expect_equal(
    torch_tensor(1)$cuda(),
    torch_tensor(1)$cpu()
  ))
})
