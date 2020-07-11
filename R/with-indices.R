torch_max_pool1d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, 
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool1d_with_indices(self, kernel, stride, padding, dilation, 
                                        ceil_mode)
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool2d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, 
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool2d_with_indices(self, kernel_size, stride, padding, dilation,
                                        ceil_mode)
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool2d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(), 
                                              padding = 0, dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool2d_with_indices_out(out, indices, self, kernel_size, stride, padding,
                                     dilation, ceil_mode)
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool3d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, 
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool3d_with_indices(self, kernel_size, stride, padding, dilation,
                                        ceil_mode)
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool3d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(), 
                                              padding = 0, dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool3d_with_indices_out(out, indices, self, kernel_size, stride, 
                                            padding, dilation, ceil_mode)
  out[[2]]$add_(1L, 1L)
  out
}