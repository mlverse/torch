variable_list <- R6::R6Class(
  classname = "torch_variable_list",
  public = list(
    ptr = NULL,
    initialize = function(x, ptr = NULL) {
      if (!is.null(ptr)) {
        self$ptr <- ptr
        return(NULL)
      }

      self$ptr <- cpp_torch_variable_list(lapply(x, function(x) x$ptr))
    },
    to_r = function() {
      x <- cpp_variable_list_to_r_list(self$ptr)
      lapply(x, function(x) Tensor$new(ptr = x))
    }
  )
)

torch_variable_list <- function(x) {
  variable_list$new(x)
}

is_torch_variable_list <- function(x) {
  inherits(x, "torch_variable_list")
}
