# Stack is the data structure used by the torch JIT to pass values.

Stack <- R6::R6Class(
  classname = "Stack",
  public = list(
    ptr = NULL,
    initialize = function(ptr = NULL) {
      if (is.null(ptr))
        self$ptr <- cpp_stack_new()
      else
        self$ptr <- ptr
    },
    push_back = function(x) {
      if (is_torch_tensor(x))
        cpp_stack_push_back_Tensor(self$ptr, x$ptr)
      else if (is.integer(x) && length(x) == 1)
        cpp_stack_push_back_int64_t(self$ptr, x)
      
      self
    },
    to_r = function() {
      r <- cpp_stack_to_r(self$ptr)
      out <- list()
      for (i in seq_along(r)) {
        if (r[[i]][[2]] == "Tensor")
          out[[i]] <- Tensor$new(ptr = r[[i]][[1]])
        else if (r[[i]][[2]] == "Int")
          out[[i]] <- r[[i]][[1]]
        else
          runtime_error("Stack contains unsupported types.")
      }
      out
    },
    at = function(i) {
      self$to_r()[[i]]
    }
  )
)