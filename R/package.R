#' @useDynLib torchpkg
#' @importFrom Rcpp sourceCpp
NULL

.generator_null <- NULL

.onAttach <- function(libname, pkgname) {
  if (!install_exists() && interactive()) {
    packageStartupMessage("You need to install libtorch in order to use torch.\n")
    ans <- readline("Do you want to download it now? ~100Mb (yes/no)")
    if (ans == "yes" | ans == "y")
      lantern_install()
    
    if (install_exists()) {
      packageStartupMessage("Torch was successfully installed.")
      packageStartupMessage("Please restart your R session now.")
    }
      
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



