test_that("local_autocast works", {
  
  x <- torch_randn(5, 5, dtype = torch_float32())
  y <- torch_randn(5, 5, dtype = torch_float32())
  
  foo <- function(x, y) {
    local_autocast(device = "cpu")
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
    w
  }
  
  out <- foo(x, y)
  expect_equal(out$dtype$.type(), "BFloat16")
  
  a <- torch_mm(x, out$float())
  expect_true(a$dtype == torch_float())
  
})

test_that("with autocast works", {
  
  x <- torch_randn(5, 5, dtype = torch_float32())
  y <- torch_randn(5, 5, dtype = torch_float32())
  with_autocast(device_type="cpu", {
    z <- torch_mm(x, y)
    w <- torch_mm(z, x)
  })
  
  expect_equal(w$dtype$.type(), "BFloat16")
  a <- torch_mm(x, w$float())
  expect_true(a$dtype == torch_float())
  
})
