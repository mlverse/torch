expect_optim_works <- function(optim, defaults) {
  
  w_true <- torch_randn(10, 1)
  x <- torch_randn(1000, 10)
  y <- torch_mm(x, w_true)
  
  loss <- function(y, y_pred) {
    torch_mean(
      (y - y_pred)^2
    )
  }
  
  w <- torch_randn(10, 1, requires_grad = TRUE)
  defaults[["params"]] <- list(w)
  opt <- do.call(optim, defaults)
  
  fn <- function() {
    opt$zero_grad()
    y_pred <- torch_mm(x, w)
    l <- loss(y, y_pred)
    l$backward()
    l
  }
  
  initial_value <- fn()
  
  iterate <- function() {
    for (i in seq_len(200)) {
      opt$step(fn)
    }  
  }
  
  profvis::profvis({
    iterate()
  })
    

  expect_true(as_array(fn()) <= as_array(initial_value)/2)
}