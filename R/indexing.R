.d <- list(
  
  `:` = function(x, y) {
    
    if (inherits(x, "slice")) {
      x$step <- y
      return(x)
    }
    
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