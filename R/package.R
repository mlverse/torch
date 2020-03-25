#' @useDynLib torchr
#' @importFrom Rcpp sourceCpp
NULL

.generator_null <- NULL

.onLoad <- function(libname, pkgname){
  
  if (lantern_installed()) {
    lantern_start() 
    .generator_null <<- torch_generator()
    .generator_null$set_current_seed(seed = abs(.Random.seed[1]))
  }
  
}

.onUnload <- function(libpath) {
  
}