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

# make sure all errors are handled
call_entries <- which(grepl(".Call\\(", exports_content))
call_skips <- paste(
  "_torch_cpp_lantern_init",
  "_torch_cpp_lantern_test",
  "_torch_cpp_lantern_has_error",
  "_torch_cpp_lantern_last_error",
  "_torch_cpp_lantern_error_clear",
  sep = "|")
for (i in call_entries) {
  if (grepl(call_skips, exports_content[[i]])) next
  exports_content[[i]] <- gsub(".Call\\(", "cpp_handle_error(.Call(", exports_content[[i]])
  exports_content[[i]] <- paste0(exports_content[[i]], ")")
}

writeLines(exports_content, exports_path)
