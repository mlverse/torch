QScheme <- R6::R6Class(
  classname = "torch_qscheme",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    print = function() {
      cat("torch_", cpp_qscheme_to_string(self$ptr), sep = "")
    }
  )
)

#' Creates the corresponding Scheme object
#'
#' @rdname torch_qscheme
#' @name torch_qscheme
#' @concept tensor-attributes
NULL

#' @rdname torch_qscheme
#' @export
torch_per_channel_affine <- function() {
  QScheme$new(cpp_torch_per_channel_affine())
}

#' @rdname torch_qscheme
#' @export
torch_per_tensor_affine <- function() {
  QScheme$new(cpp_torch_per_tensor_affine())
}

#' @rdname torch_qscheme
#' @export
torch_per_channel_symmetric <- function() {
  QScheme$new(cpp_torch_per_channel_symmetric())
}

#' @rdname torch_qscheme
#' @export
torch_per_tensor_symmetric <- function() {
  QScheme$new(cpp_torch_per_tensor_symmetric())
}

#' Checks if an object is a QScheme
#'
#' @param x object to check
#'
#' @export
is_torch_qscheme <- function(x) {
  inherits(x, "torch_qscheme")
}
