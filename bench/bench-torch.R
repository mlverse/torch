library(torch)
Sys.setenv(KMP_DUPLICATE_LIB_OK=TRUE)

x <- torch_randn(1000, 784)
w <- torch_randn(784, 10)

bench::mark(
  a = torch_mm(x, w),
  b = x$mm(w),
  c = Tensor$new(ptr = cpp_torch_namespace_mm_self_Tensor_mat2_Tensor(x$ptr, w$ptr)),
  min_time = 1
)