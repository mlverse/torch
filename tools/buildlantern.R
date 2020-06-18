
if (!require(fs, quietly = TRUE)) {
  install.packages("fs")
}

if (dir.exists("lantern")) {
  cat("Building lantern .... \n")
  
  if (!fs::dir_exists("lantern/build"))
    fs::dir_create("lantern/build")
  
  withr::with_dir("lantern/build", {
    system("cmake ..")
    system("cmake --build . --target lantern --config Release")  
  })

  # copy lantern
  source("R/lantern_sync.R")
  lantern_sync(TRUE)  
  
  # download torch
  source("R/lantern_install.R")
  install_torch(path = normalizePath("deps/"))
  
  # copy deps to inst
  if (fs::dir_exists("inst/deps"))
    fs::dir_delete("inst/deps/")
  
  fs::dir_copy("deps/", new_path = "inst/deps/")
}
  

