test_that("Forking doesn't dealock", {
  skip_on_os(c("windows", "mac"))
  
  out <- callr::r(function() {
    library(torch)
    testfun <- function (x) {
      lstm <- nn_lstm(50, 50)
      in_data <- torch_tensor(matrix(rnorm(50),1), torch_float())
      out_data <- torch_tensor(array(rnorm(2500), c(1,50,50)), torch_float())
      out_pred <- lstm(in_data)[[1]]
      loss <- nnf_mse_loss(out_pred, out_data)
      loss$backward()
      loss$item()
    }
    parallel::mclapply(1:2, testfun)
  })
  
  expect_true(length(out), 2)
  expect_true(is.numeric(out[[1]]))
})
