torch_dtype <- R6::R6Class(
  classname = "torch_dtype", 
  public = list(
    ptr = NULL,
    initialize = function(ptr = NULL) {
      self$ptr <- ptr
    },
    print = function() {
      cat("torch_", cpp_dtype_to_string(self$ptr), sep = "")
    }
  ),
  active = list(
    is_floating_point = function() {
      if (cpp_dtype_to_string(self$ptr) %in% c("Float", "Double", "Half"))
        TRUE
      else
        FALSE
    }
  )
)

#' @export
torch_float32 <- function() torch_dtype$new(cpp_torch_float32())
#' @export
torch_float <- function() torch_dtype$new(cpp_torch_float32())

#' @export
torch_float64 <- function() torch_dtype$new(cpp_torch_float64())
#' @export
torch_double <- function() torch_dtype$new(cpp_torch_float64())

#' @export
torch_float16 <- function() torch_dtype$new(cpp_torch_float16())
#' @export
torch_half <- function() torch_dtype$new(cpp_torch_float16())

#' @export
torch_uint8 <- function() torch_dtype$new(cpp_torch_uint8())

#' @export
torch_int8 <- function() torch_dtype$new(cpp_torch_int8())

#' @export
torch_int16 <- function() torch_dtype$new(cpp_torch_int16())
#' @export
torch_short <- function() torch_dtype$new(cpp_torch_int16())

#' @export
torch_int32 <- function() torch_dtype$new(cpp_torch_int32())
#' @export
torch_int <- function() torch_dtype$new(cpp_torch_int32())

#' @export
torch_int64 <- function() torch_dtype$new(cpp_torch_int64())
#' @export
torch_long <- function() torch_dtype$new(cpp_torch_int64())

#' @export
torch_bool <- function() torch_dtype$new(cpp_torch_bool())

#' @export
`==.torch_dtype` <- function(e1, e2) {
  cpp_dtype_to_string(e1$ptr) == cpp_dtype_to_string(e2$ptr)
}

is_torch_dtype <- function(x) {
  inherits(x, "torch_dtype")
}
