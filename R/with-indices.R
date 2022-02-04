torch_max_pool1d_with_indices <- function(self, kernel_size, stride = list(), padding = 0,
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool1d_with_indices(
    self, kernel_size, stride, padding, dilation,
    ceil_mode
  )
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool2d_with_indices <- function(self, kernel_size, stride = list(), padding = 0,
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool2d_with_indices(
    self, kernel_size, stride, padding, dilation,
    ceil_mode
  )
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool2d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(),
                                              padding = 0, dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool2d_with_indices_out(
    out, indices, self, kernel_size, stride, padding,
    dilation, ceil_mode
  )
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool3d_with_indices <- function(self, kernel_size, stride = list(), padding = 0,
                                          dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool3d_with_indices(
    self, kernel_size, stride, padding, dilation,
    ceil_mode
  )
  out[[2]]$add_(1L, 1L)
  out
}

torch_max_pool3d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(),
                                              padding = 0, dilation = 1, ceil_mode = FALSE) {
  out <- .torch_max_pool3d_with_indices_out(
    out, indices, self, kernel_size, stride,
    padding, dilation, ceil_mode
  )
  out[[2]]$add_(1L, 1L)
  out
}

torch_max <- function(self, dim, other, keepdim = FALSE) {
  o <- do.call(.torch_max, as.list(environment()))
  if (is.list(o) && length(o) == 2) {
    o[[2]]$add_(1L, 1L)
  }
  o
}

torch_min <- function(self, dim, other, keepdim = FALSE) {
  args <- as.list(environment())
  o <- do.call(.torch_min, args)
  if (is.list(o) && length(o) == 2) {
    o[[2]]$add_(1L, 1L)
  }
  o
}

torch_argmax <- function(self, dim = NULL, keepdim = FALSE) {
  o <- .torch_argmax(self, dim = dim, keepdim = keepdim)
  o$add_(1L, 1L)
  o
}

torch_argmin <- function(self, dim = NULL, keepdim = FALSE) {
  o <- .torch_argmin(self, dim = dim, keepdim = keepdim)
  o$add_(1L, 1L)
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

#' @rdname torch_argsort
torch_argsort <- function(self, dim = -1L, descending = FALSE) {
  .torch_argsort(self = self, dim = dim, descending = descending)$add_(1L, 1L)
}

torch_cross_entropy_loss <- function(self, target, weight = list(),
                                     reduction = torch_reduction_mean(),
                                     ignore_index = -100L) {
  target <- target$sub(1L, 1L)
  .torch_cross_entropy_loss(
    self = self, target = target, weight = weight,
    reduction = reduction, ignore_index = ignore_index
  )
}

torch_sort <- function(self, dim = -1L, descending = FALSE, stable) {
  if (missing(stable)) {
    out <- .torch_sort(self = self, dim = dim, descending = descending)
  } else {
    out <- .torch_sort(self = self, dim = dim, descending = descending, stable = stable)
  }
  out[[2]]$add_(1L)
  out
}
