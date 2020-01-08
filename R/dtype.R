torch_dtype <- R6::R6Class(
  classname = "torch_dtype", 
  public = list(
    initialize = function(ptr = NULL, f_ptr = NULL) {
      if (!is.null(f_ptr))
        private$f_ptr <- f_ptr
      else if (!is.null(ptr))
        private$ptr_ <- ptr
      else
        stop("Specify one of f_ptr or ptr")
    },
    print = function() {
      cat(cpp_torch_dtype_to_string(self$ptr()))
    },
    ptr = function() {
      if (is.null(private$ptr_))
        private$ptr_ <- private$f_ptr()
      
      private$ptr_
    },
    is_floating_point = function(x) {
      
      if (cpp_torch_dtype_to_string(self$ptr()) %in% c("float32", "float64", "float16"))
        TRUE
      else
        FALSE
    }
  ),
  private = list(
    f_ptr = NULL,
    ptr_ = NULL
  )
)

#' @export
torch_float32 <- torch_dtype$new(f_ptr = cpp_torch_float32)
#' @export
torch_float <- torch_dtype$new(f_ptr = cpp_torch_float32)

#' @export
torch_float64 <- torch_dtype$new(f_ptr = cpp_torch_float64)
#' @export
torch_double <- torch_dtype$new(f_ptr = cpp_torch_float64)

#' @export
torch_float16 <- torch_dtype$new(f_ptr = cpp_torch_float16)
#' @export
torch_half <- torch_dtype$new(f_ptr = cpp_torch_float16)

#' @export
torch_uint8 <- torch_dtype$new(f_ptr = cpp_torch_uint8)

#' @export
torch_int8 <- torch_dtype$new(f_ptr = cpp_torch_int8)

#' @export
torch_int16 <- torch_dtype$new(f_ptr = cpp_torch_int16)
#' @export
torch_short <- torch_dtype$new(f_ptr = cpp_torch_int16)

#' @export
torch_int32 <- torch_dtype$new(f_ptr = cpp_torch_int32)
#' @export
torch_int <- torch_dtype$new(f_ptr = cpp_torch_int32)

#' @export
torch_int64 <- torch_dtype$new(f_ptr = cpp_torch_int64)
#' @export
torch_long <- torch_dtype$new(f_ptr = cpp_torch_int64)

#' @export
torch_bool <- torch_dtype$new(f_ptr = cpp_torch_bool)


