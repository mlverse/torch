.globals <- new.env(parent = emptyenv())
.globals$lantern_started <- FALSE

lantern_start <- function(reload = FALSE) {
  if (!torch_is_installed()) {
    stop("Torch is not installed, please run 'install_torch()'.")
  }

  if (.globals$lantern_started && !reload) {
    return()
  }

  cpp_lantern_init(file.path(torch_install_path(), "lib"))

  log_enabled <- as.integer(Sys.getenv("TORCH_LOG", "0"))
  cpp_lantern_configure(log_enabled)

  .globals$lantern_started <- TRUE
}

