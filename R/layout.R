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

#' @export
torch_strided <- function() torch_layout$new(cpp_torch_strided())

#' @export
torch_sparse_coo <- function() torch_layout$new(cpp_torch_sparse_coo())
