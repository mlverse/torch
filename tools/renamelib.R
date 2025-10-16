libs_path <- getwd()

for (lib in dir(libs_path, pattern = "torch\\.")) {
  file.copy(file.path(libs_path, lib),
            file.path(libs_path, gsub("torch", "torchpkg", lib)),
            overwrite = TRUE)

  # Uncomment CRAN release
  # unlink(file.path(libs_path, lib))
}


exports_path <- try({
  normalizePath(file.path(libs_path, "..", "R", "RcppExports.R"), mustWork = TRUE)
}, silent = TRUE)

# load_all
if (inherits(exports_path, "try-error")) {
  exports_path <- try({
    normalizePath(file.path(libs_path, "R", "RcppExports.R"), mustWork = TRUE)
  }, silent = TRUE)
}

if (inherits(exports_path, "try-error")) {
  stop("Could not patch RcppExports.R, looked for files in: \n",
       normalizePath(file.path(libs_path, "..", "R", "RcppExports.R"), mustWork = FALSE), "\n",
       normalizePath(file.path(libs_path, "R", "RcppExports.R"), mustWork = FALSE)
  )
}

exports_content <- readLines(exports_path)

exports_content <- gsub("PACKAGE = 'torch'", "PACKAGE = 'torchpkg'", exports_content)

writeLines(exports_content, exports_path)
