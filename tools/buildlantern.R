
if (!require(fs, quietly = TRUE)) {
  install.packages("fs")
}

if (!require(fs))
  stop("fs was not correctly installed?")

if (dir.exists("lantern")) {
  cat("Building lantern .... \n")
  
  if (!fs::dir_exists("lantern/build"))
    fs::dir_create("lantern/build")
  
  withr::with_dir("lantern/build", {
    system("cmake ..")
    system("cmake --build . --target lantern --config Release --parallel 8")  
  })

  # copy lantern
  source("R/lantern_sync.R")
  lantern_sync(TRUE)  
  
  # download torch
  source("R/install.R")
  install_torch(path = normalizePath("inst/"), load = FALSE)
}
  

