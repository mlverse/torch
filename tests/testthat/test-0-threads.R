# theses tests must run the before any torch op.

test_that("can set threads", {
  skip_on_os("windows")
  skip_on_os("mac")

  old <- torch_get_num_interop_threads()
  torch_set_num_interop_threads(6)
  expect_equal(torch_get_num_interop_threads(), 6)

  old <- torch_get_num_threads()
  torch_set_num_threads(6)
  expect_equal(torch_get_num_threads(), 6)
  torch_set_num_threads(old)
})
