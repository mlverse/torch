test_that("can compare tensors with testthat edition 3", {
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
  expect_failure(expect_equal(
    torch_tensor(1),
    torch_tensor(2)
  ))

  skip_if_no_cuda()

  expect_failure(expect_equal(
    torch_tensor(1)$cuda(),
    torch_tensor(1)$cpu()
  ))
})
