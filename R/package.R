#' @useDynLib torchr
#' @importFrom Rcpp sourceCpp
NULL

.generator_null <- NULL

.onLoad <- function(libname, pkgname){
  
  if (!lantern_installed() && Sys.getenv("INSTALL_TORCH", unset = 0) == 1) {
    packageStartupMessage("Installing torch to ", lantern_install_path(), " ...")
    lantern_install()
    if (lantern_installed())
      packageStartupMessage("Successfully installed torch!\n")
  }
    
  if (!lantern_installed() && interactive()) {
    packageStartupMessage("You need to install libtorch in order to use torchr.\n")
    ans <- readline("Do you want to download it now? ~100Mb (yes/no)")
    if (ans == "yes" | ans == "y")
      lantern_install()
  }
    
  if (lantern_installed()) {
    lantern_start() 
    .generator_null <<- torch_generator()
    .generator_null$set_current_seed(seed = abs(.Random.seed[1]))
  }
  
}

.onUnload <- function(libpath) {
  
}