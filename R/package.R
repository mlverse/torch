#' @useDynLib torchpkg
#' @importFrom Rcpp sourceCpp
NULL

.generator_null <- NULL

.onAttach <- function(libname, pkgname) {
}

.onLoad <- function(libname, pkgname){
  install_success <- TRUE
  if (!install_exists() && Sys.getenv("INSTALL_TORCH", unset = 1) != 0) {
    install_success <- tryCatch({
      install_torch()
      TRUE
    }, error = function(e) {
      warning("Failed to install Torch, manually run install_torch(). ", e$message, call. = FALSE)
      FALSE
    })
  }
    
  if (install_exists() && install_success && Sys.getenv("LOAD_TORCH", unset = 1) != 0) {
    lantern_start() 
    .generator_null <<- torch_generator()
    .generator_null$set_current_seed(seed = abs(.Random.seed[1]))
  }
}

.onUnload <- function(libpath) {
  
}



