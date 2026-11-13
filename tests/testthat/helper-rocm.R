# Helpers for the ROCm/HIP tests (test-rocm.R).
#
# These tests need an installed ROCm libtorch, a working KFD device and the
# R package built with the kit, so they only run when explicitly requested
# (like TORCH_TEST_CUDA for the existing CUDA tests). The self-hosted GPU job
# sets TORCH_TEST_ROCM=1.

skip_if_rocm_not_requested <- function() {
  if (Sys.getenv("TORCH_TEST_ROCM", unset = "0") != "1") {
    skip("Set TORCH_TEST_ROCM=1 to run the ROCm/HIP tests.")
  }
  if (.Platform$OS.type != "unix" || Sys.info()[["sysname"]] != "Linux") {
    skip("ROCm is only supported on Linux.")
  }
}

rocm_is_available <- function() {
  lib <- tryCatch(torch::torch_install_path(), error = function(e) NULL)
  !is.null(lib) && torch:::rocm_bundle_is_installed(lib)
}

skip_if_no_rocm_bundle <- function() {
  if (!rocm_is_available()) {
    skip("The installed libtorch is not a ROCm/HIP bundle.")
  }
}
