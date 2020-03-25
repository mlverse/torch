#' @export
`+.torch_tensor` <- function(e1, e2) {
  torch_add(e1, e2)
}

#' @export
`-.torch_tensor` <- function(e1, e2) {
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
  if (is.name(s)) {
    return(x)
  } else if (length(s) == 1)
    return(torch_select(x, dim = dim, index = s))
}

#' @export
`[.torch_tensor` <- function(x, ...) {
  slices <- rlang::dots_list(..., .preserve_empty = TRUE, .ignore_empty = "none")
  d <- dim(x)
  for (dim in seq_along(slices)) {
    x <- slice_dim(x, dim - 1 - (length(d) - length(dim(x))), slices[[dim]])
  }
  x
}


