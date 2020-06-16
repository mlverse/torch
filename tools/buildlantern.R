
if (!require(fs, quietly = TRUE)) {
  install.packages("fs")
}

if (dir.exists("lantern")) {
  cat("Building lantern .... \n")
  
  if (!fs::dir_exists("lantern/build"))
    fs::dir_create("lantern/build")
  
  withr::with_dir("lantern/build", {
    system("cmake ..")
    system("cmake --build .")  
  })

  # copy lantern
  source("R/lantern_sync.R")
  lantern_sync(TRUE)  
  
  # download torch
  source("R/lantern_install.R")
  lantern_install(check_installed = TRUE)
  
  # copy deps to inst
  fs::dir_delete("inst/deps/")
  fs::dir_copy("deps/", new_path = "inst/deps/")
}
  

