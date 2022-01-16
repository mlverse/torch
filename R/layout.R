torch_layout <- R6::R6Class(
  classname = "torch_layout",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    print = function() {
      cat("torch_", cpp_layout_to_string(self$ptr), sep = "")
    }
  )
)

#' Creates the corresponding layout
#'
#' @name torch_layout
#' @rdname torch_layout
NULL

#' @rdname torch_layout
#' @export
torch_strided <- function() torch_layout$new(cpp_torch_strided())

#' @rdname torch_layout
#' @export
torch_sparse_coo <- function() torch_layout$new(cpp_torch_sparse())

#' Check if an object is a torch layout.
#'
#' @param x object to check
#'
#' @export
is_torch_layout <- function(x) {
  inherits(x, "torch_layout")
}
