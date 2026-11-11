# test_torch_rocm.R -- {torch} smoke test for a ROCm/HIP build (make_rocm_kit.sh)
#
# Run from the {torch} tree root:   source("tools/test_torch_rocm.R")
# or headless:                      Rscript tools/test_torch_rocm.R
#
# Last change: v17 -- refuse to test a CPU/CUDA libtorch bundle (mixing it with
# the ROCm liblantern is an ABI chimera whose symptoms look like R-patch bugs),
# and fix the device-RNG assertion (torch_equal() returns a plain R logical).
#
# v16 -- run the matmul probe in an isolated Rscript: a missing Tensile kernel
# makes rocBLAS abort the process (SIGABRT), which tryCatch() cannot catch and
# which used to kill the whole smoke test.
#
# Expected on a gfx900-class card (Vega10, ROCm 6.4 wheel):
#   [ok]   device detection, allocations, ATen element-wise kernels,
#          deterministic device RNG, the whole torch::cuda_* inventory
#   [info] matmul -> the wheel's rocBLAS ships no Tensile kernels for gfx900
#          (pre-gfx908 cards); see tools/make_rocm_kit.sh --doctor
#
# Every check is collected and reported; the script stops (when needed) only
# at the END, so you always get the full picture first.

Sys.setenv(TORCH_INSTALL = "0")   # never prompt to download mid-test

score <- list(ok = 0L, warn = 0L, fail = 0L)
say <- function(status, msg) {
  tag <- switch(status, ok = "ok", warn = "WARN", fail = "FAIL", info = "info")
  if (!is.null(score[[status]])) score[[status]] <<- score[[status]] + 1L
  cat(sprintf("[%s] %s\n", tag, msg))
}

cat("torch R package :", as.character(packageVersion("torch")), "\n")
lantern_v <- tryCatch(
  as.character(torch::lantern_version()),
  error = function(e) tryCatch(
    as.character(getExportedValue(asNamespace("torch"), "lantern_version")),
    error = function(e2) NA_character_))
cat("lantern         :",
    if (is.na(lantern_v)) "(version API not exported by this torch build)"
    else lantern_v, "\n")

# --- 1. package load ------------------------------------------------------------
loaded <- tryCatch({
  suppressPackageStartupMessages(library(torch))
  TRUE
}, error = function(e) {
  say("fail", paste("library(torch) failed:", conditionMessage(e)))
  FALSE
})
if (!loaded) {
  stop("torch could not load -- run the kit (tools/make_rocm_kit.sh), then ",
       "torch::install_torch() with TORCH_URL=<tree>/build-lantern/libtorch and ",
       "LANTERN_URL=<tree>/torch-rocm-kit/lantern, and restart R.")
}

# The INSTALLED libtorch bundle must be the ROCm/HIP one. liblantern is linked
# against build-lantern/libtorch, so dropping it next to a CPU/CUDA libtorch
# makes the loader pull libtorch_hip from the build tree while libtorch_cpu
# comes from the installed package: same sonames, different builds -> virtual
# calls land on the wrong method (field signature: cuda_synchronize() reports
# an impossible "device=-80" while everything element-wise still works).
lib_dir <- file.path(torch::torch_install_path(), "lib")
has_hip <- length(list.files(lib_dir, pattern = "^libtorch_hip\\.so")) > 0L
if (!has_hip) {
  say("fail", sprintf(
    "installed libtorch is not the ROCm bundle (no libtorch_hip.so in %s)",
    lib_dir))
  stop("reinstall the ROCm libtorch from the kit, then restart R:\n",
       "  Sys.setenv(TORCH_URL   = '<tree>/build-lantern/libtorch',\n",
       "             LANTERN_URL = '<tree>/torch-rocm-kit/lantern')\n",
       "  torch::install_torch(reinstall = TRUE)")
}
say("info", sprintf("libtorch bundle: %s (ROCm/HIP)", lib_dir))

# --- 2. device detection ----------------------------------------------------------
avail <- tryCatch(torch::cuda_is_available(), error = function(e) {
  say("fail", paste("cuda_is_available() failed:", conditionMessage(e)))
  NA
})
if (!isTRUE(avail)) {
  if (identical(avail, FALSE)) say("fail", "cuda_is_available() = FALSE")
  stop("CUDA/HIP device unavailable -- check rocminfo, the video/render ",
       "groups, and tools/make_rocm_kit.sh --doctor.")
}
say("ok", "cuda_is_available() = TRUE (HIP device enumerated by KFD)")
n_dev <- tryCatch(torch::cuda_device_count(), error = function(e) NA_integer_)
if (!is.na(n_dev)) say("info", sprintf("cuda_device_count() = %d", n_dev))

# --- 3. allocation + ATen kernels on device ---------------------------------------
x <- tryCatch(torch::torch_randn(1024L, device = "cuda"), error = function(e) {
  say("fail", paste("torch_randn(cuda) failed:", conditionMessage(e)))
  NULL
})
if (!is.null(x)) {
  say("ok", "allocation + ATen kernel (randn) executed on device")
  s <- tryCatch((x + 1)$sum()$item(), error = function(e) {
    say("fail", paste("add+sum on device failed:", conditionMessage(e)))
    NULL
  })
  if (!is.null(s)) say("ok", sprintf("element-wise op + reduction OK (sum = %.1f)", s))
}

# --- 4. device RNG ----------------------------------------------------------------
if (!is.null(x)) {
  det <- tryCatch({
    torch::torch_manual_seed(42)
    a <- torch::torch_rand(4L, device = "cuda")
    torch::torch_manual_seed(42)
    b <- torch::torch_rand(4L, device = "cuda")
    isTRUE(torch::torch_equal(a, b))
  }, error = function(e) {
    say("fail", paste("device RNG failed:", conditionMessage(e)))
    FALSE
  })
  if (isTRUE(det)) say("ok", "device RNG: manual_seed + torch_rand is deterministic")
}

# --- 5. torch::cuda_* inventory (the kit enables the real implementations) -------
if (!is.null(x)) {
  say("info", "torch::cuda_* inventory:")
  apis <- list(
    c("cuda_current_device",        function() torch::cuda_current_device()),
    c("cuda_get_device_capability", function() torch::cuda_get_device_capability()),
    c("cuda_runtime_version",       function() torch::cuda_runtime_version()),
    c("cuda_memory_stats",          function() torch::cuda_memory_stats()),
    c("cuda_memory_summary",        function() invisible(torch::cuda_memory_summary())),
    c("cuda_empty_cache",           function() torch::cuda_empty_cache()),
    c("cuda_synchronize",           function() torch::cuda_synchronize()),
    c("cuda_get_rng_state",         function() torch::cuda_get_rng_state())
  )
  for (a in apis) {
    nm <- a[[1]]; fn <- a[[2]]
    res <- tryCatch({ fn(); "ok" }, error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("could not find function", msg)) "not exported"
      else paste("error:", sub("\n.*", "", msg))
    })
    if (identical(res, "ok")) {
      say("ok", sprintf("%s() works", nm))
    } else if (identical(res, "not exported")) {
      say("warn", sprintf("%s() is not exported by this torch build (update the package)", nm))
    } else {
      hint <- ""
      if (grepl("libcaffe2_nvrtc", res)) {
        hint <- " -- rerun the kit (it stages the libcaffe2_nvrtc -> hiprtc bridge)"
      } else if (grepl("only supported on CUDA", res)) {
        hint <- " -- rebuild liblantern with the kit (the Cuda.cpp patch was missing)"
      } else if (grepl("device=-", res)) {
        hint <- " -- rebuild liblantern with the kit (the R/cuda.R patch was missing)"
      }
      say("fail", sprintf("%s() failed: %s%s", nm, res, hint))
    }
  }
}

# --- 6. matmul -> expected to fail on gfx900 (rocBLAS without TensileLibrary) -----
# A missing Tensile kernel does NOT raise an R error: rocBLAS prints
# "Cannot read ... TensileLibrary.dat ... for GPU arch : gfx900" and aborts
# the whole process (SIGABRT -> exit status 134). tryCatch() cannot intercept
# a signal, so the probe runs in an isolated Rscript; a crash is then reported
# here instead of killing the session. The child also disables core dumps
# (a torch process core file can be several GB).
matmul_probe <- function() {
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
  list(status = if (is.null(status)) 0L else status, output = out)
}
if (!is.null(x)) {
  mm <- matmul_probe()
  if (mm$status == 0L && any(grepl("MATMUL_OK", mm$output, fixed = TRUE))) {
    say("ok", "matmul: OK (surprising on this card!)")
  } else {
    detail <- grep("rocBLAS error|HIP error|Error", mm$output, value = TRUE)
    detail <- if (length(detail)) sub("\n.*", "", detail[[1]]) else "no detail"
    how <- if (mm$status > 128L) {
      sprintf("isolated R process killed by signal %d", mm$status - 128L)
    } else {
      sprintf("isolated R process exited with status %d", mm$status)
    }
    if (grepl("TensileLibrary", detail, fixed = TRUE)) {
      say("info", sprintf("matmul: unavailable (expected on gfx900) -- %s; %s",
                          how, detail))
    } else {
      say("warn", sprintf("matmul: probe failed unexpectedly -- %s; %s",
                          how, detail))
    }
  }
}

# --- summary ------------------------------------------------------------------------
cat(sprintf("\nSummary: %d ok / %d warnings / %d failures\n",
            score$ok, score$warn, score$fail))
if (score$fail > 0L) {
  stop("torch-on-ROCm smoke test FAILED -- paste the full output plus ",
       "tools/make_rocm_kit.sh --doctor into the issue.")
}
cat("If everything above is [ok] (matmul may be [info]): the full\n",
    "R -> liblantern -> libtorch -> HIP -> gfx900 chain works and the\n",
    "torch::cuda_* API is functional on the ROCm wheel.\n", sep = "")
invisible(NULL)
