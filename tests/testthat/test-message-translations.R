test_that("R-level error messages are correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_error(
                          torch_scalar_tensor(torch_randn(8, 2, 7)),
                        regexp = "les valeurs doivent être de longueur 1",
                        fixed = TRUE
                      )
  )
})

test_that("R-level warning messages are correctly translated in FR", {
  x <- torch_randn(2, requires_grad = TRUE)
  y <- torch_randn(1)
  b <- (x^y)$sum()
  y$add_(1)
  withr::with_language(lang = "fr",
                       expect_warning(
                         with_detect_anomaly({
                           b$backward()
                         }),
                        regexp = "Ce mode ne doit être activé qu'en cas de débogage",
                        fixed = TRUE
                      )
  )
})

