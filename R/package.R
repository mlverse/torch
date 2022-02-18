#' @useDynLib torchpkg
#' @importFrom Rcpp sourceCpp
NULL

globalVariables(c("..", "self", "private", "N"))

.generator_null <- NULL
.compilation_unit <- NULL

.onAttach <- function(libname, pkgname) {
}

.onLoad <- function(libname, pkgname) {
  cpp_torch_namespace__store_main_thread_id()

  install_success <- TRUE
  autoinstall <- interactive() ||
    "JPY_PARENT_PID" %in% names(Sys.getenv()) ||
    identical(getOption("jupyter.in_kernel"), TRUE)

  if (!install_exists() && Sys.getenv("TORCH_INSTALL", unset = 2) != 0 &&
    (autoinstall || Sys.getenv("TORCH_INSTALL", unset = 2) == "1")) {
    install_success <- tryCatch(
      {
        install_torch()
        TRUE
      },
      error = function(e) {
        warning("Failed to install Torch, manually run install_torch().\n ", e$message, call. = FALSE)
        FALSE
      }
    )
  }

  if (install_exists() && install_success && Sys.getenv("TORCH_LOAD", unset = 1) != 0) {
    # in case init fails aallow user to restart session rather than blocking install
    tryCatch(
      {
        lantern_start()
        cpp_set_lantern_allocator(getOption("torch.threshold_call_gc", 4000L))
        register_lambda_function_deleter()

        # .generator_null is no longer used. set the option `torch.old_seed_behavior=TRUE` to use it.
        .generator_null <<- torch_generator()
        .generator_null$set_current_seed(seed = sample(1e5, 1))

        .compilation_unit <<- cpp_jit_compilation_unit()
      },
      error = function(e) {
        warning("Torch failed to start, restart your R session to try again. ", e$message, call. = FALSE)
        FALSE
      }
    )
  }
}

.onUnload <- function(libpath) {

}

release_bullets <- function() {
  c(
    "Create the cran/ branch and update the branch variable",
    "Uncomment the indicated line in the .RBuildignore file"
  )
}
