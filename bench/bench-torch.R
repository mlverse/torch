library(torch)
Sys.setenv(KMP_DUPLICATE_LIB_OK=TRUE)

x <- torch_randn(1000, 784)
w <- torch_randn(784, 10)

bench::mark(
  torch_mm(x, w),
  x$mm(w)
)
