
.torch_index <- function() {
  x <- torch_randn(10)
  i <- torch_tensor(c(1,2), dtype = torch_long())
  
  ind <- cpp_torch_tensor_index_new()
  cpp_torch_tensor_index_append_tensor(ind, i$ptr)
  
  p <- cpp_torch_tensor_index(x$ptr, ind)
  Tensor$new(ptr = p)
}
