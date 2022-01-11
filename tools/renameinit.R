libs_path <- getwd()

exports_path <- try({
  normalizePath(file.path(libs_path, "..", "src", "RcppExports.cpp"), mustWork = TRUE)
}, silent = TRUE)

# load_all
if (inherits(exports_path, "try-error")) {
  exports_path <- try({
    normalizePath(file.path(libs_path, "src", "RcppExports.cpp"), mustWork = TRUE)
  }, silent = TRUE)
}

if (inherits(exports_path, "try-error")) {
  stop("Could not patch RcppExports.R, looked for files in: \n",
       normalizePath(file.path(libs_path, "..", "R", "RcppExports.R"), mustWork = FALSE), "\n",
       normalizePath(file.path(libs_path, "R", "RcppExports.R"), mustWork = FALSE)
  )
}

exports_content <- readLines(exports_path)

exports_content <- gsub("R_init_torch\\(", "R_init_torchpkg(", exports_content)

writeLines(exports_content, exports_path)
