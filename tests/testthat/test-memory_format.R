context("memory-format")

test_that("Can create memory formats", {
  expect_s3_class(torch_contiguous_format(), "torch_memory_format")
  expect_s3_class(torch_preserve_format(), "torch_memory_format")
  expect_s3_class(torch_channels_last_format(), "torch_memory_format")
})

test_that("Can compare memory formats", {
  expect_true(torch_contiguous_format() == torch_contiguous_format())
  expect_false(torch_preserve_format() == torch_contiguous_format())
})
