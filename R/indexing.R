.d <- list(
  
  `:` = function(x, y) {
    
    if (inherits(x, "slice")) {
      x$step <- y
      return(x)
    }
    
    # end should be inclusive
    if (y == -1)
      y <- .Machine$integer.max
    else if (y < -1)
      y <-  y + 1
    
    e <- list(start = ifelse(x > 0, x - 1, x), end = y, step = 1)
    attr(e, "class") <- "slice"
    e
  },
  
  N = .Machine$integer.max, 
  newaxis = NULL,
  
  `..` = structure(list(), class = "fill")
)

tensor_slice <- function(tensor, ..., drop = TRUE) {
  Tensor$new(ptr = Tensor_slice(tensor$ptr, environment(), drop = drop, mask = .d))
}

#' @export
`[.torch_tensor` <- function(x, ..., drop = TRUE) {
  tensor_slice(x, ..., drop = drop)
}

tensor_slice_put_ <- function(tensor, ..., value) {
  
  if (length(value) > 1 && is.atomic(value))
    value <- torch_tensor(value)
  
  Tensor_slice_put(tensor$ptr, environment(), value, mask = .d)
}

#' @export
`[<-.torch_tensor` <- function(x, ..., value) {
  tensor_slice_put_(x, ..., value = value)
  invisible(x)
}