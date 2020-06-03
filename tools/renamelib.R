current_path <- "C:/Users/Administrator/code/torch/src"

exports_path <- normalizePath(file.path(current_path, "..", "R", "RcppExports.R"))
exports_content <- readLines(exports_path)

exports_content <- gsub("PACKAGE = 'torch'", "PACKAGE = 'torchpkg'", exports_content)
writeLines(exports_content, exports_path)
