test_that("Forking doesn't deadlock", {
  skip_on_os(c("windows", "mac"))
  
  out <- callr::r(timeout = 30, function() {
    library(torch)
    testfun <- function (x) {
      lstm <- nn_lstm(50, 50)
      in_data <- torch_randn(1,50,50)
      out_data <- torch_randn(1,50,50)
      out_pred <- lstm(in_data)[[1]]
      loss <- nnf_mse_loss(out_pred, out_data)
      loss$backward()
      loss$item()
    }
    parallel::mclapply(1:2, testfun)
  })
  
  expect_equal(length(out), 2)
  expect_true(is.numeric(out[[1]]))
})
