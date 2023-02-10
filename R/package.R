#' @useDynLib torchpkg, .registration = TRUE
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
  
  is_interactive <- interactive() ||
    "JPY_PARENT_PID" %in% names(Sys.getenv()) ||
    identical(getOption("jupyter.in_kernel"), TRUE)
  
  # we only autoinstall if it has not explicitly disabled by setting 
  # TORCH_INSTALL = 0
  autoinstall <- is_interactive && (Sys.getenv("TORCH_INSTALL", unset = 2) != 0)
  
  # We can also auto install if TORCH_INSTALL is requested with TORCH_INSTALL=1
  autoinstall <- autoinstall || (Sys.getenv("TORCH_INSTALL", unset = 2) == "1")
  
  # we only autoinstall if installation doesn't yet exist.
  autoinstall <- autoinstall && (!torch_is_installed())
  
  if (autoinstall) {
    install_success <- tryCatch(
      {
        cli::cli_alert_info("Additional software needs to be {.strong downloaded} and {.strong installed} for torch to work correctly.")
        check_can_autoinstall() # this errors if it's not possible to autoinstall for that system
        # in interactive environments we want to ask the user for permission to
        # download and install stuff. That's not necessary otherwise because the
        # user has explicitly asked for installation with `TORCH_INSTALL=1`.
        if (is_interactive) { 
          get_confirmation() # this will error of response is not true.  
        }
        install_torch()
        TRUE
      },
      error = function(e) {
        msg <- if (is.character(e$message)) e$message else "Unknown error."
        cli::cli_warn(c(
          i = "Failed to install torch, manually run {.fn install_torch}",
          x = msg
        ), parent = e)
        FALSE
      }
    )
  }

  if (torch_is_installed() && install_success && Sys.getenv("TORCH_LOAD", unset = 1) != 0) {
    # in case init fails aallow user to restart session rather than blocking install
    tryCatch(
      {
        lantern_start()
        cpp_set_lantern_allocator(getOption("torch.threshold_call_gc", 4000L))
        cpp_set_cuda_allocator_allocator_thresholds(
          getOption("torch.cuda_allocator_reserved_rate", 0.2),
          getOption("torch.cuda_allocator_allocated_rate", 0.8),
          getOption("torch.cuda_allocator_allocated_reserved_rate", 0.8)
        )
        register_lambda_function_deleter()

        # .generator_null is no longer used. set the option `torch.old_seed_behavior=TRUE` to use it.
        .generator_null <<- torch_generator()
        .generator_null$set_current_seed(seed = sample(1e5, 1))

        .compilation_unit <<- cpp_jit_compilation_unit()
      },
      error = function(e) {
        msg <- if (is.character(e$message)) e$message else "Unknown error."
        cli::cli_warn(c(
          i = "torch failed to start, restart your R session to try again.",
          i = "You might need to reinstall torch using {.fn install_torch}",
          x = msg
        ), parent = e)
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

get_confirmation <-  function() {
  response <- utils::askYesNo(msg = "Do you want to continue?")
  if (is.na(response) || (!response)) {
    stop("Aborted.", call. = FALSE)
  }
  TRUE
}

check_can_autoinstall <- function() {
  if (!grepl("x86_64", R.version$arch)) {
    if (!grepl("darwin", R.version$os)) {
      cli::cli_abort(c(
        "Currently only {.code x86_64} systems are supported for automatic installation. ",
        i = "You can manually compile LibTorch for you architecture following instructions in {.url https://github.com/pytorch/pytorch#from-source}",
        i = "You can then use {.fn install_torch_from_file} to install manually."
      ))
    }
  }
  TRUE
}
