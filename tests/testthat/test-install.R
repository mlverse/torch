test_that("file installation ", {
  expect_false(is.null(get_install_libs_url()$libtorch))
  expect_false(is.null(get_install_libs_url()$liblantern))
})
