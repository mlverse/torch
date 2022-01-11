Scalar <- R7Class(
  classname = "torch_scalar",
  public = list(
    ptr = NULL,
    initialize = function(x, ptr = NULL) {
      if (!is.null(ptr)) {
        return(ptr)
      }

      cpp_torch_scalar(x)
    },
    to_r = function() {
      type <- self$type

      if (type == torch_double()) {
        f <- cpp_torch_scalar_to_double
      } else if (type == torch_float()) {
        f <- cpp_torch_scalar_to_float
      } else if (type == torch_bool()) {
        f <- cpp_torch_scalar_to_bool
      } else if (type == torch_int()) {
        f <- cpp_torch_scalar_to_int
      } else if (type == torch_long()) {
        f <- cpp_torch_scalar_to_int
      }

      f(self$ptr)
    },
    print = function() {
      cat("torch_scalar\n")
    }
  ),
  active = list(
    type = function() {
      torch_dtype$new(ptr = cpp_torch_scalar_dtype(self$ptr))
    },
    ptr = function() {
      self
    }
  )
)

torch_scalar <- function(x) {
  Scalar$new(x)
}

is_torch_scalar <- function(x) {
  inherits(x, "torch_scalar")
}
