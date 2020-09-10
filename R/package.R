#' @useDynLib torchpkg
#' @importFrom Rcpp sourceCpp
NULL

globalVariables(c("..", "self", "private", "N"))

.generator_null <- NULL

.onAttach <- function(libname, pkgname) {
}

.onLoad <- function(libname, pkgname){
  install_success <- TRUE
  autoinstall <- interactive() || "JPY_PARENT_PID" %in% names(Sys.getenv())
  
  if (!install_exists() && Sys.getenv("TORCH_INSTALL", unset = 2) != 0 && 
      (autoinstall || Sys.getenv("TORCH_INSTALL", unset = 2) == "1")) {
    install_success <- tryCatch({
      install_torch()
      TRUE
    }, error = function(e) {
      warning("Failed to install Torch, manually run install_torch(). ", e$message, call. = FALSE)
      FALSE
    })
  }
    
  if (install_exists() && install_success && Sys.getenv("TORCH_LOAD", unset = 1) != 0) {
    # in case init fails aallow user to restart session rather than blocking install
    tryCatch({
      lantern_start() 
      .generator_null <<- torch_generator()
      .generator_null$set_current_seed(seed = abs(.Random.seed[1]))
    }, error = function(e) {
      warning("Torch failed to start, restart your R session to try again. ", e$message, call. = FALSE)
      FALSE
    })
  }
}

.onUnload <- function(libpath) {
  
}



