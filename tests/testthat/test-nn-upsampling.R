test_that("upsample works", {
  input <- torch_arange(start = 1, end = 4, dtype = torch_float())$view(c(1, 1, 2, 2))
  out <- nn_upsample(scale_factor = c(2), mode = "nearest")(input)

  expect_equal_to_r(out[1, 1, 1:2, 1:2], matrix(rep(1, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 3:4, 1:2], matrix(rep(3, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 1:2, 3:4], matrix(rep(2, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 3:4, 3:4], matrix(rep(4, 4), ncol = 2))

  out <- nn_upsample(scale_factor = c(2, 2), mode = "nearest")(input)
  expect_equal_to_r(out[1, 1, 1:2, 1:2], matrix(rep(1, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 3:4, 1:2], matrix(rep(3, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 1:2, 3:4], matrix(rep(2, 4), ncol = 2))
  expect_equal_to_r(out[1, 1, 3:4, 3:4], matrix(rep(4, 4), ncol = 2))

  out <- nn_upsample(scale_factor = 2, mode = "bilinear")(input)
  expect_equal(out$shape, c(1, 1, 4, 4))

  out <- nn_upsample(scale_factor = 2, mode = "bilinear", align_corners = TRUE)(input)
  expect_equal(out$shape, c(1, 1, 4, 4))
  expect_equal_to_r(out[1, 1, 1, 1], 1)
})

test_that("errors are raised for upsample", {
  input <- torch_arange(start = 1, end = 4, dtype = torch_float())$view(c(1, 1, 2, 2))
  expect_error(
    out <- nn_upsample(scale_factor = c(2), size = c(4, 4), mode = "nearest")(input),
    class = "value_error"
  )

  expect_error(
    out <- nn_upsample(scale_factor = c(2, 2, 2), mode = "nearest")(input),
    class = "value_error"
  )

  expect_error(
    out <- nn_upsample()(input),
    class = "value_error"
  )
})
