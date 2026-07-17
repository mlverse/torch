cpp_extension_package <- local({
  namespace <- NULL

  function() {
    if (!is.null(namespace)) return(namespace)

    package <- testthat::test_path("..", "cpp-extension")
    library <- file.path(tempdir(), "torch-cpp-extension-library")
    dir.create(library, recursive = TRUE, showWarnings = FALSE)

    status <- system2(
      file.path(R.home("bin"), "R"),
      c("CMD", "INSTALL", "--clean", "--no-multiarch",
        paste0("--library=", shQuote(library)),
        shQuote(package)),
      stdout = TRUE,
      stderr = TRUE
    )
    install_status <- attr(status, "status")
    if (is.null(install_status)) install_status <- 0L
    testthat::expect_equal(
      install_status,
      0L,
      info = paste(status, collapse = "\n")
    )

    namespace <<- loadNamespace("torchextensiontest", lib.loc = library)
    library(
      "torchextensiontest",
      lib.loc = library,
      character.only = TRUE,
      warn.conflicts = FALSE
    )
    namespace
  }
})
