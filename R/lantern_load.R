.globals <- new.env(parent = emptyenv())
.globals$lantern_started <- FALSE

lantern_default <- function() {
  "1.4.0"
}

lantern_start <- function(version = lantern_default(), reload = FALSE) {
  if (!install_exists()) {
    stop("Torch is not installed, please run 'install_torch()'.")
  }

  if (.globals$lantern_started && !reload) {
    return()
  }

  cpp_lantern_init(file.path(install_path(), "lib"))

  log_enabled <- as.integer(Sys.getenv("TORCH_LOG", "0"))
  cpp_lantern_configure(log_enabled)

  .globals$lantern_started <- TRUE
}

lantern_test <- function() {
  lantern_start()

  cpp_lantern_test()
}
