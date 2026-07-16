.globals <- new.env(parent = emptyenv())
.globals$lantern_started <- FALSE

load_cudatoolkit_libs <- function() {
  cuda_ver <- cuda_version_from_cudatoolkit()
  if (is.null(cuda_ver)) return(invisible(FALSE))

  pkg_name <- paste0("cuda", cuda_ver)
  if (!requireNamespace(pkg_name, quietly = TRUE)) return(invisible(FALSE))

  lib_path <- getExportedValue(pkg_name, "lib_path")()
  if (!dir.exists(lib_path)) return(invisible(FALSE))

  if (is_windows()) {
    # Add the lib path so Windows can find the DLLs
    Sys.setenv(PATH = paste(lib_path, Sys.getenv("PATH"), sep = ";"))
    libs <- list.files(lib_path, pattern = "\\.dll$", full.names = TRUE)
  } else {
    libs <- list.files(lib_path, pattern = "\\.so(\\.[0-9.]+)?$", full.names = TRUE)
    # Only load real files, skip symlinks to avoid double-loading
    libs <- libs[!nzchar(Sys.readlink(libs))]
  }

  for (lib in libs) {
    tryCatch(
      dyn.load(lib, local = FALSE, now = FALSE),
      error = function(e) NULL
    )
  }

  invisible(TRUE)
}

lantern_start <- function(reload = FALSE) {
  if (!torch_is_installed()) {
    runtime_error("Torch is not installed, please run 'install_torch()'.")
  }

  if (.globals$lantern_started && !reload) {
    return()
  }

  load_cudatoolkit_libs()

  lib_path <- file.path(torch_install_path(), "lib")
  if (is_windows()) {
    # cuDNN 9 is split across several DLLs. cuDNN's lazy sub-DLL load (e.g.
    # cudnn_graph64_9.dll) does not find the install lib dir unless it is on
    # PATH, so cuDNN-backed CUDA ops otherwise fail with "Could not locate
    # cudnn_graph64_9.dll" even though the DLL is present here. Prepend lib_path
    # (once) so those loads resolve.
    path_dirs <- strsplit(Sys.getenv("PATH"), ";", fixed = TRUE)[[1]]
    if (!any(normalizePath(path_dirs, winslash = "/", mustWork = FALSE) ==
             normalizePath(lib_path, winslash = "/", mustWork = FALSE))) {
      Sys.setenv(PATH = paste(lib_path, Sys.getenv("PATH"), sep = ";"))
    }
  }
  cpp_lantern_init(lib_path)

  log_enabled <- as.integer(Sys.getenv("TORCH_LOG", "0"))
  cpp_lantern_configure(log_enabled)

  .globals$lantern_started <- TRUE
}

