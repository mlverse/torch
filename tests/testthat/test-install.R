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

test_that("is_rocm_version identifies rocm strings", {
  expect_true(torch:::is_rocm_version("rocm6.4"))
  expect_false(torch:::is_rocm_version("12.8"))
  expect_false(torch:::is_rocm_version("cpu"))
  expect_false(torch:::is_rocm_version(""))
})

test_that("check_supported_version accepts supported CUDA versions", {
  expect_no_error(torch:::check_supported_version("12.8", c("12.6", "12.8")))
})

test_that("check_supported_version aborts on unsupported CUDA version", {
  expect_error(torch:::check_supported_version("11.0", c("12.6", "12.8")))
})

test_that("check_supported_rocm_version_linux accepts rocm6.4", {
  expect_no_error(torch:::check_supported_rocm_version_linux("rocm6.4"))
})

test_that("check_supported_rocm_version_linux aborts on unknown rocm version", {
  expect_error(torch:::check_supported_rocm_version_linux("rocm5.0"))
})

test_that("cuda_version_from_cudatoolkit returns NULL for CUDA=rocm6.4 without package", {
  withr::with_envvar(list(CUDA = "rocm6.4", TORCH_CUDATOOLKIT = ""), {
    result <- torch:::cuda_version_from_cudatoolkit()
    expect_null(result)
  })
})

test_that("cuda_version_from_cudatoolkit does not error for CUDA=rocm6.4", {
  withr::with_envvar(list(CUDA = "rocm6.4", TORCH_CUDATOOLKIT = ""), {
    expect_no_error(torch:::cuda_version_from_cudatoolkit())
  })
})

test_that("TORCH_CUDATOOLKIT + CUDA=rocm6.4 conflict is detected", {
  skip_if_not(is_linux() || is_windows())
  withr::with_envvar(list(TORCH_CUDATOOLKIT = "12.8", CUDA = "rocm6.4"), {
    expect_error(torch:::cuda_version_from_cudatoolkit(), "conflicts with")
  })
})

test_that("rocm_version_from_system_linux returns NULL when ROCm not found", {
  skip_if_not(is_linux())
  withr::with_envvar(list(ROCM_PATH = "", ROCM_HOME = ""), {
    result <- suppressMessages(torch:::rocm_version_from_system_linux())
    expect_null(result)
  })
})

test_that("load_cudatoolkit_libs does not look for cudarocm6.4 package", {
  with_mocked_bindings(
    cuda_version_from_cudatoolkit = function() "rocm6.4",
    .package = "torch",
    {
      expect_no_error(torch:::load_cudatoolkit_libs())
    }
  )
})