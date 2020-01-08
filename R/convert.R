#' @useDynLib torch
#' @importFrom Rcpp sourceCpp
NULL

#' Create torch Tensor from R object
#'
#' @param x an R vector, matrix or array.
#' @param dtype dtype
#' @param device device
#' @param requires_grad requires_grad
#'
#' @note it uses the R type when creating the tensor.
#'
#' @examples
#' tensor_from_r(1:10)
#' tensor_from_r(array(runif(8), dim = c(2, 2, 2)))
#' tensor_from_r(matrix(c(TRUE, FALSE), nrow = 3, ncol = 4))
#' @export
tensor_from_r <- function(x, dtype = NULL, device = NULL, requires_grad = FALSE) {

  dimension <- dim(x)

  if (is.null(dimension)) {
    dimension <- length(x)
  }

  `torch::Tensor`$dispatch(tensor_from_r_(x, rev(dimension), dtype, device, requires_grad))
}

#' Creates a torch tensor.
#'
#' @param x an R object or a torch tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' x <- tensor(1:10)
#' x
#'
#' y <- tensor(x, dtype = "double")
#' y
#' @export
tensor <- function(x, ...) {
  UseMethod("tensor", x)
}

#' @export
tensor.default <- function(x, dtype = NULL, device = NULL, requires_grad = FALSE) {
  tensor_from_r(x, dtype, device, requires_grad)
}

#' @export
tensor.tensor <- function(x, dtype = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(tensor_(x$pointer, dtype, device, requires_grad))
}

#' Tensor casting
#'
#' Casts an object with class [tensor] to an R atomic vector, matrix or array.
#'
#' @param x tensor object to be casted to an R array.
#' @seealso [as.matrix.tensor()]
#' @examples
#' x <- tensor(array(1:8, dim = c(2, 2, 2)))
#' as.array(x)
#' @export
as.array.tensor <- function(x) {
  x$as_vector()
}

#' Casts a 2d tensor to a matrix.
#'
#' @param x tensor object
#' @seealso [as.array.tensor()]
#'
#' @examples
#' x <- tensor(as.matrix(mtcars))
#' as.matrix(x)
#' @export
as.matrix.tensor <- function(x) {
  as.matrix(x$as_vector())
}



