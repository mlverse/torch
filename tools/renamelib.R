libs_path <- getwd()

for (lib in dir(libs_path, pattern = "torch\\.")) {
  file.copy(file.path(libs_path, lib), file.path(libs_path, gsub("torch", "torchpkg", lib)))
}

exports_path <- normalizePath(file.path(libs_path, "..", "R", "RcppExports.R"))
exports_content <- readLines(exports_path)

exports_content <- gsub("PACKAGE = 'torch'", "PACKAGE = 'torchpkg'", exports_content)
writeLines(exports_content, exports_path)
