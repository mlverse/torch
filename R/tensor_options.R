TensorOptions <- R6::R6Class(
  classname = "torch_tensor_options",
  public = list(
    ptr = NULL,
    initialize = function(dtype = NULL, layout = NULL, device = NULL, 
                          requires_grad = NULL, pinned_memory = NULL, ptr = NULL) {
      
      if (!is.null(ptr)) {
        self$ptr <- ptr
        return(NULL)
      }
      
      self$ptr <- cpp_torch_tensor_options(dtype$ptr, layout$ptr, device$ptr, requires_grad,
                                           pinned_memory)
    },
    print = function() {
      cat("torch_tensor_options")
      cpp_torch_tensor_options_print(self$ptr)
    }
  )
)

torch_tensor_options <- function(dtype = NULL, layout = NULL, device = NULL, 
                                 requires_grad = NULL, pinned_memory = NULL) {
  TensorOptions$new(dtype, layout, device, requires_grad, pinned_memory)
}

is_torch_tensor_options <- function(x) {
  inherits(x, "torch_tensor_options")
}

as_torch_tensor_options <- function(l) {
 do.call(torch_tensor_options, l) 
}