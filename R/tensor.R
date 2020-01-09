Tensor <- R6::R6Class(
  classname = "torch_tensor",
  public = list(
    ptr = NULL,
    initialize = function(data = NULL, dtype = NULL, device = NULL, requires_grad = FALSE, 
                          pin_memory = FALSE, ptr = NULL) {
      
      if (!is.null(ptr)) {
        self$ptr <- ptr
        return(NULL)
      }
      
      # infer dtype from data
      if (is.null(dtype)) {
        
        if (is.integer(data)) {
          dtype <- torch_int()
        } else if (is.double(data)) {
          dtype <- torch_double()
        } else if (is.logical(data)) {
          dtype <- torch_bool()
        }
        
      }
      
      options <- torch_tensor_options(dtype = dtype, device = device, 
                                      requires_grad = requires_grad, 
                                      pinned_memory = pin_memory)
      
      
      dimension <- dim(data)
      
      if (is.null(dimension)) {
        dimension <- length(data)
      }
      
      
      self$ptr <- cpp_torch_tensor(data, rev(dimension), options$ptr)
    },
    print = function() {
      cat(sprintf("torch_tensor \n"))
      tensor_print_(self$ptr)
      invisible(self)
    }
  )
)

#' @export
torch_tensor <- function(data, dtype = NULL, device = NULL, requires_grad = FALSE, 
                         pin_memory = FALSE) {
  Tensor$new(data, dtype, device, requires_grad, pin_memory)
}

as_array <- function(x) {
  UseMethod("as_array", x)
}

as_array.torch_tensor <- function(x) {
  a <- cpp_as_array(x$ptr)
  
  if (length(a$dim) <= 1L) {
    out <- a$vec
  } else if (length(a$dim) == 2L) {
    out <- t(matrix(a$vec, ncol = a$dim[1], nrow = a$dim[2]))
  } else {
    out <- aperm(array(a$vec, dim = rev(a$dim)), seq(length(a$dim), 1))
  }
  
  out
}