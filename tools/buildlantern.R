
if (!require(fs, quietly = TRUE)) {
  install.packages("fs")
}

if (dir.exists("lantern")) {
  cat("Building lantern .... \n")
  
  if (!fs::dir_exists("lantern/build"))
    fs::dir_create("lantern/build")
  
  withr::with_dir("lantern/build", {
    if (!.Platform$OS.type == "windows")
      system("cmake ..")
    else
      system("cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..")
    system("cmake --build . --target lantern --config Release --parallel 8")  
  })

  # copy lantern
  source("R/lantern_sync.R")
  lantern_sync(TRUE)  
  
  # download torch
  source("R/install.R")
  install_torch(path = normalizePath("deps/"), load = FALSE)
  
  # copy deps to inst
  if (fs::dir_exists("inst/deps"))
    fs::dir_delete("inst/deps/")
  
  fs::dir_copy("deps/", new_path = "inst/deps/")
}
  

