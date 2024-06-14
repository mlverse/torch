test_that("R-level message for nn_conv1d padding_mode value_error() get translated in french", {
  testthat::skip_on_ci()
  testthat::skip_on_cran()
  withr::with_language(lang = "fr",
                       expect_error(
                         nn_conv1d(3, 3, 2, 1, padding_mode = "bad_padding"),
                         regexp = "‘padding_mode’ doit être pris parmi "
                       )
  )
})

test_that("R-level message for runtime_error() get translated in french", {
  testthat::skip_on_ci()
  testthat::skip_on_cran()
  withr::with_language(lang = "fr",
                       expect_error(
                         jit_tuple(c(7,3)),
                         regexp = "L’argument ‘x’ doit être une liste."
                       )
  )
})
