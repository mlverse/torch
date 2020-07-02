context("qscheme")

test_that("Can create QSchemes", {
  expect_s3_class(torch_per_channel_affine(), "torch_qscheme")
  expect_s3_class(torch_per_channel_symmetric(), "torch_qscheme")
  expect_s3_class(torch_per_tensor_affine(), "torch_qscheme")
  expect_s3_class(torch_per_channel_symmetric(), "torch_qscheme")
})
