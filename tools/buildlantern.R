if (dir.exists("lantern")) {
  cat("Building lantern .... \n")

  dir.create("lantern/build", showWarnings = FALSE, recursive = TRUE)

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


