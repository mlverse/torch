test_that("experimental Tensor methods are up to date", {
  python <- Sys.which("python3")
  skip_if(!nzchar(python), "python3 is not available")
  generator <- test_path("..", "..", "tools", "generate-experimental-tensor.py")
  skip_if(!file.exists(generator), "code generator is not available")

  output <- system2(
    python,
    c(generator, "--check"),
    stdout = TRUE,
    stderr = TRUE
  )

  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  expect_equal(status, 0L, info = paste(output, collapse = "\n"))
})

test_that("experimental namespace names match the source operations", {
  python <- Sys.which("python3")
  skip_if(!nzchar(python), "python3 is not available")
  regression_tests <- test_path(
    "..", "..", "tools", "test-generate-experimental-tensor.py"
  )
  skip_if(!file.exists(regression_tests), "code generator tests are not available")

  output <- system2(python, regression_tests, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  expect_equal(status, 0L, info = paste(output, collapse = "\n"))
})
