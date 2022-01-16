test_that("nnf_normalize", {
  x <- torch_tensor(c(1, 1, 0, 0))

  expect_error(nnf_normalize(x))
  expect_equal_to_tensor(
    nnf_normalize(x, dim = 1),
    torch_tensor(c(0.7071, 0.7071, 0, 0)),
    tolerance = 1e-5
  )

  out <- torch_empty(4)
  nnf_normalize(x, dim = 1, out = out)
  expect_equal_to_tensor(
    out,
    torch_tensor(c(0.7071, 0.7071, 0, 0)),
    tolerance = 1e-5
  )
})

test_that("nnf_local_response_norm", {
  signal_2d <- torch_randn(32, 5, 24, 24)
  signal_4d <- torch_randn(16, 5, 7, 7, 7, 7)

  output_2d <- nnf_local_response_norm(signal_2d, size = 2)
  output_4d <- nnf_local_response_norm(signal_4d, size = 2)

  expect_length(output_2d$size(), 4)
  expect_length(output_4d$size(), 6)
})
