TensorList <- R7Class(
  classname = "torch_tensor_list",
  public = list(
    ptr = NULL,
    initialize = function(x, ptr = NULL) {
      
      if (!is.null(ptr)) {
        return(ptr)
      }
      
      cpp_torch_tensor_list(lapply(x, function(x) x$ptr))
    },
    to_r = function() {
      x <- cpp_tensor_list_to_r_list(self$ptr)
      lapply(x, function(x) Tensor$new(ptr = x))
    }
  ),
  active = list(
    ptr = function() {
      self
    }
  )
)

torch_tensor_list <- function(x) {
  TensorList$new(x)
}

is_torch_tensor_list <- function(x) {
  inherits(x, "torch_tensor_list")
}