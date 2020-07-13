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

torch_max <- function(self, dim, other, keepdim = FALSE) {
  o <- do.call(.torch_max, as.list(environment()))
  if (length(o) == 2)
    o[[2]]$add_(1L, 1L)
  o
}

torch_min <- function(self, dim, other, keepdim = FALSE) {
  o <- do.call(.torch_min, as.list(environment()))
  if (length(o) == 2)
    o[[2]]$add_(1L, 1L)
  o
}

torch_nll_loss <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  target <- target$sub(1L, 1L)
  .torch_nll_loss(self, target, weight, reduction, ignore_index)
}

torch_nll_loss2d <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  target <- target$sub(1L, 1L)
  .torch_nll_loss2d(self, target, weight, reduction, ignore_index)
}