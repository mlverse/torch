source_cpp_test <- function(file, env = parent.frame()) {
  cache_dir <- file.path(tempdir(), "torch-test-cpp-cache")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  path <- testthat::test_path("..", "cpp", file)
  Rcpp::sourceCpp(
    path,
    env = env,
    cacheDir = cache_dir,
    rebuild = FALSE,
    showOutput = FALSE,
    verbose = FALSE
  )

  invisible(path)
}
