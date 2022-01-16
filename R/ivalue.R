
#' Adds the 'jit_tuple' class to the input
#'
#' Allows specifying that an output or input must be considered a jit
#' tuple and instead of a list or dictionary when tracing.
#'
#' @param x the list object that will be converted to a tuple.
#'
#' @export
jit_tuple <- function(x) {
  if (!is.list(x)) {
    runtime_error("Argument 'x' must be a list.")
  }

  class(x) <- c(class(x), "jit_tuple")
  x
}

#' Adds the 'jit_scalar' class to the input
#'
#' Allows disambiguating length 1 vectors from scalars when passing
#' them to the jit.
#'
#' @param x a length 1 R vector.
#'
#' @export
jit_scalar <- function(x) {
  if (!rlang::is_scalar_atomic(x)) {
    runtime_error("Argument 'x' must be scalar atomic.")
  }

  class(x) <- c(class(x), "jit_scalar")
  x
}
