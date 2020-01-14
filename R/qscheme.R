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

#' @export
torch_per_channel_affine <- function() {
  QScheme$new(cpp_torch_per_channel_affine())
}

#' @export
torch_per_tensor_affine <- function() {
  QScheme$new(cpp_torch_per_tensor_affine())
}

#' @export
torch_per_channel_symmetric <- function() {
  QScheme$new(cpp_torch_per_channel_symmetric())
}

#' @export
torch_per_tensor_symmetric <- function() {
  QScheme$new(cpp_torch_per_tensor_symmetric())
}