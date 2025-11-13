test_that("file installation ", {
  expect_false(is.null(get_install_libs_url()$libtorch))
  expect_false(is.null(get_install_libs_url()$liblantern))
})

test_that("installer provides an explicit message when target folder has read-only access", {
  skip_if_not(is_linux())
  withr::with_envvar(new = c("TORCH_INSTALL_DEBUG" = TRUE, "TORCH_HOME" = "/dev/mem"),  
    expect_error(
      torch:::inst_path(check_writable=TRUE),
      regexp = "cannot write into configured",
      fixed = TRUE
    )
  )
  
})