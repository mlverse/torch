#' @useDynLib torchpkg
#' @importFrom Rcpp sourceCpp
NULL

.generator_null <- NULL

.onAttach <- function(libname, pkgname) {
  if (!install_exists() && interactive()) {
    install_torch()
  }
}

.onLoad <- function(libname, pkgname){
  if (!install_exists() && Sys.getenv("INSTALL_TORCH", unset = 0) == 1) {
    install_torch()
  }
    
  if (install_exists()) {
    lantern_start() 
    .generator_null <<- torch_generator()
    .generator_null$set_current_seed(seed = abs(.Random.seed[1]))
  }
  
}

.onUnload <- function(libpath) {
  
}



