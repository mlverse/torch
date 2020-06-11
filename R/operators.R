#' @export
`+.torch_tensor` <- function(e1, e2) {
  if (missing(e2))
    e2 <- torch_zeros_like(e1)
  
  torch_add(e1, e2)
}

#' @export
`-.torch_tensor` <- function(e1, e2) {
  if (missing(e2))
    e2 <- torch_zeros_like(e1)
  
  torch_sub(e1, e2)
}

#' @export
`*.torch_tensor` <- function(e1, e2) {
  torch_mul(e1, e2)
}

#' @export
`/.torch_tensor` <- function(e1, e2) {
  torch_div(e1, e2)
}

#' @export
`^.torch_tensor` <- function(e1, e2) {
  torch_pow(e1, e2)
}

#' @export
`>=.torch_tensor` <- function(e1, e2) {
  torch_ge(e1, e2)
}

#' @export
`>.torch_tensor` <- function(e1, e2) {
  torch_gt(e1, e2)
}

#' @export
`<=.torch_tensor` <- function(e1, e2) {
  torch_le(e1, e2)
}

#' @export
`<.torch_tensor` <- function(e1, e2) {
  torch_lt(e1, e2)
}

#' @export
`==.torch_tensor` <- function(e1, e2) {
  torch_eq(e1, e2)
}

#' @export
`!=.torch_tensor` <- function(e1, e2) {
  torch_ne(e1, e2)
}

#' @export
dim.torch_tensor <- function(x) {
  cpp_tensor_dim(x$ptr)
}

#' @export
length.torch_tensor <- function(x) {
  prod(dim(x))
}

slice_dim <- function(x, dim, s) {
  if (length(s) == 1 && all(is.na(s)))
    return(x)
  
  if (length(s) == 1)
    return(torch_select(x, dim = dim, index = ifelse(s > 0, s - 1, s)))
  
  if (inherits(s, "slice"))
    return(torch_slice(x, dim, s$start, s$end, s$step))
}

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
  
  `..` = structure(list(), class = "fill")
)

#' @export
`[.torch_tensor` <- function(x, ...) {
  slices <- rlang::enquos(..., .ignore_empty = "none")
  
  slices <- lapply(slices, function(x) {
    if(rlang::is_missing(rlang::quo_get_expr(x))) 
      NA 
    else 
      lazyeval::lazy_eval(x, data = .d)
  })
  
  d <- dim(x)
  
  
  if (length(slices) <= length(d)) {
    if (inherits(slices[[1]], "fill")) {
      slices <- slices[-1]
      a <- as.list(rep(NA, length(d) - length(slices)))
      slices <- append(a, slices)
    } else if (inherits(slices[[length(slices)]], "fill")) {
      slices <- slices[-length(slices)]
      a <- as.list(rep(NA, length(d) - length(slices)))
      slices <- append(slices, a)
    }
  }
  
  if (length(slices) != length(d))
    stop("incorrect number of dimensions. Specified " , length(slices), " should be ",
         length(d), ".")
  
  for (dim in seq_along(slices)) {
    x <- slice_dim(x, dim - 1 - (length(d) - length(dim(x))), slices[[dim]])
  }
  x
}


