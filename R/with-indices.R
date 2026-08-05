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

# `ignore_index` names a target *value*, and is compared against the target inside libtorch after
# `to_index_tensor()` has made that target 0 based. It therefore has to undergo the same
# translation, or it selects the class one to the left of the one that was asked for.
# Values that cannot be a 1 based class index are left alone: the default `-100`, and any other
# negative sentinel, is meant to match no target at all and does so in either convention.
to_index_ignore_index <- function(ignore_index) {
  if (is.null(ignore_index)) {
    return(ignore_index)
  }
  if (ignore_index == 0) {
    value_error("`ignore_index` is a 1 based class index, so it cannot be 0. Use a negative value (e.g. the default -100) for a target value that never occurs.")
  }
  if (ignore_index >= 1) ignore_index - 1L else ignore_index
}

torch_nll_loss <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  target <- to_index_tensor(target)
  .torch_nll_loss(self, target, weight, reduction, to_index_ignore_index(ignore_index))
}

torch_nll_loss2d <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  target <- to_index_tensor(target)
  .torch_nll_loss2d(self, target, weight, reduction, to_index_ignore_index(ignore_index))
}

#' @rdname torch_argsort
torch_argsort <- function(self, dim = -1L, descending = FALSE) {
  .torch_argsort(self = self, dim = dim, descending = descending)$add_(1L, 1L)
}

torch_cross_entropy_loss <- function(self, target, weight = list(),
                                     reduction = torch_reduction_mean(),
                                     ignore_index = -100L) {
  target <- to_index_tensor(target)
  .torch_cross_entropy_loss(
    self = self, target = target, weight = weight,
    reduction = reduction, ignore_index = to_index_ignore_index(ignore_index)
  )
}

torch_nll_loss_nd <- function(self, target, weight = list(), reduction = torch_reduction_mean(), 
                               ignore_index = -100L) {
  target <- to_index_tensor(target)
  .torch_nll_loss_nd(
    self = self, target = target, weight = weight,
    reduction = reduction,
    ignore_index = to_index_ignore_index(ignore_index)
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

torch_bincount <- function(self, weights = list(), minlength = 0L) { 
  .torch_bincount(self = to_index_tensor(self), weights = weights,
                  minlength = minlength)
}
