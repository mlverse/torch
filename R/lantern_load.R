.globals <- new.env(parent = emptyenv())
.globals$lantern_started <- FALSE

load_cudatoolkit_libs <- function() {
  cuda_ver <- cuda_version()
  if (is.null(cuda_ver) || cuda_ver == "cpu") return(invisible(FALSE))

  pkg_name <- paste0("cuda", cuda_ver)
  if (!requireNamespace(pkg_name, quietly = TRUE)) return(invisible(FALSE))

  lib_path <- getExportedValue(pkg_name, "lib_path")()
  if (!dir.exists(lib_path)) return(invisible(FALSE))

  libs <- list.files(lib_path, pattern = "\\.so(\\.[0-9.]+)?$", full.names = TRUE)
  # Only load real files, skip symlinks to avoid double-loading
  libs <- libs[!nzchar(Sys.readlink(libs))]

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
  cpp_lantern_init(file.path(torch_install_path(), "lib"))

  log_enabled <- as.integer(Sys.getenv("TORCH_LOG", "0"))
  cpp_lantern_configure(log_enabled)

  .globals$lantern_started <- TRUE
}

