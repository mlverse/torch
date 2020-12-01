test_that("simnple tracing works", {
  
  fn <- function(x) {
    torch_relu(x)
  }
  
  input <- torch_tensor(c(-1, 0, 1))
  tr_fn <- jit_trace(fn, input)
  
  expect_equal_to_tensor(tr_fn(input), fn(input))
})
