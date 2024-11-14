test_that("R-level error messages are correctly translated in FR", {
  # skip on ubuntu cuda as image as no FR lang installed
  skip_if(torch::cuda_is_available() && grepl("linux-gnu", R.version$os))
  # skip on MAC M1
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")
  withr::with_language(lang = "fr",
                       expect_error(
                          torch_scalar_tensor(torch_randn(8, 2, 7)),
                        regexp = "les valeurs doivent être de longueur 1",
                        fixed = TRUE
                      )
  )
})

test_that("R-level warning messages are correctly translated in FR", {
  # skip on ubuntu cuda as image as no FR lang installed
  skip_if(torch::cuda_is_available() && grepl("linux-gnu", R.version$os))
  # skip on MAC M1
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")
  x <- torch_randn(2, requires_grad = TRUE)
  y <- torch_randn(1)
  b <- (x^y)$sum()
  y$add_(1)
  withr::with_language(lang = "fr",
                       expect_warning(
                         nnf_mse_loss(torch_randn(5), torch_randn(5, 1)),
                        regexp = "est différente de taille de l'entrée",
                        fixed = TRUE
                      )
  )
})

