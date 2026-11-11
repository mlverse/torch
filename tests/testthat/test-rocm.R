skip_if_rocm_not_requested()

rocm_matmul_probe <- function() {
  libs <- paste(sprintf('"%s"', .libPaths()), collapse = ", ")
  code <- c(
    sprintf(".libPaths(c(%s))", libs),
    'suppressPackageStartupMessages(library(torch))',
    'a <- torch::torch_randn(64L, 64L, device = "cuda")',
    'b <- torch::torch_randn(64L, 64L, device = "cuda")',
    'torch::torch_mm(a, b)',
    'cat("\\nMATMUL_OK\\n")'
  )
  script <- tempfile("matmul-probe-", fileext = ".R")
  writeLines(code, script)
  on.exit(unlink(script), add = TRUE)
  cmd <- sprintf(
    "ulimit -c 0; exec %s --no-init-file --no-save --no-restore %s",
    shQuote(file.path(R.home("bin"), "Rscript")), shQuote(script))
  out <- suppressWarnings(
    system2("/bin/sh", c("-c", shQuote(cmd)), stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else as.integer(status), output = out)
}

test_that("the installed libtorch is a ROCm/HIP bundle", {
  lib <- torch::torch_install_path()
  expect_true(dir.exists(lib), info = lib)
  expect_true(
    file.exists(file.path(lib, "lib", "libtorch_hip.so")),
    info = "install the ROCm libtorch with make_rocm_kit.sh --install"
  )
})

test_that("the HIP device is visible", {

  skip_if_no_rocm_bundle()

  expect_true(torch::cuda_is_available())
  expect_gt(torch::cuda_device_count(), 0)
  expect_gte(torch::cuda_current_device(), 0)
})

test_that("element-wise kernels run on the HIP device", {
  skip_if_no_rocm_bundle()

  x <- torch::torch_randn(1024L, device = "cuda")
  expect_equal(x$device$type, "cuda")
  expect_equal(x$device$index, 0L)

  total <- (x + 1)$sum()$item()
  expect_type(total, "double")
  expect_true(is.finite(total))
})

test_that("device RNG is deterministic with manual_seed", {

  skip_if_no_rocm_bundle()

  torch::torch_manual_seed(42)
  a <- torch::torch_rand(4L, device = "cuda")
  torch::torch_manual_seed(42)
  b <- torch::torch_rand(4L, device = "cuda")
  expect_true(torch::torch_equal(a, b))
})

test_that("the torch::cuda_* API is functional on the ROCm build", {
  skip_if_no_rocm_bundle()

  apis <- c(
    "cuda_current_device",
    "cuda_get_device_capability",
    "cuda_runtime_version",
    "cuda_memory_stats",
    "cuda_memory_summary",
    "cuda_empty_cache",
    "cuda_synchronize",
    "cuda_get_rng_state"
  )
  for (nm in apis) {
    fn <- getExportedValue("torch", nm)
    expect_no_error(
      if (identical(nm, "cuda_memory_summary")) capture.output(fn()) else fn(),
      info = sprintf("%s() failed on the ROCm build", nm)
    )
  }
})

test_that("matmul works on the HIP device", {
  skip_if_no_rocm_bundle()

  probe <- rocm_matmul_probe()
  ok <- probe$status == 0L &&
    any(grepl("MATMUL_OK", probe$output, fixed = TRUE))
  if (ok) {
    succeed()
    return(invisible(NULL))
  }

  detail <- grep("rocBLAS error|HIP error|Error", probe$output, value = TRUE)
  detail <- if (length(detail)) sub("\n.*", "", detail[[1]]) else "no detail"
  if (grepl("TensileLibrary", detail, fixed = TRUE)) {
    skip(paste("known gfx900 limitation: rocBLAS has no Tensile kernels --",
               detail))
  }
  fail(paste0("matmul probe failed (status ", probe$status, "): ", detail))
})
