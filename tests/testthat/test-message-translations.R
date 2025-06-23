test_that("R-level error messages are correctly translated in FR", {
  # skip on ubuntu cuda as image as no FR lang installed
  skip_if(torch::cuda_is_available() && is_linux())
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
  skip_if(torch::cuda_is_available() && is_linux())
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

test_that("installer_message gets correctly translated in FR", {
  skip_if(torch::cuda_is_available() && is_linux())
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")
  withr::with_envvar(new = c("TORCH_INSTALL_DEBUG" = TRUE),  
    withr::with_language(lang = "fr",
       expect_message(
         torch:::nvcc_version_from_path(tempfile()),
        regexp = "Tentative de lancer nvcc depuis",
        fixed = TRUE
      )
  ))
})

test_that("cli_abort gets correctly translated in FR", {
  skip_if(torch::cuda_is_available() && is_linux())
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")
    withr::with_language(lang = "fr",
       expect_error(
         torch:::check_supported_version("7.3.2", c("10.1", "10.2")),
        regexp = "La version de CUDA ",
        fixed = TRUE
      )
  )
})

