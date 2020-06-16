#' @include R7.R

Tensor <- R7Class(
  classname = "torch_tensor", 
  # lock_class = FALSE,
  # lock_objects = FALSE,
  public = list(
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
          dtype <- torch_float() # default to float
        } else if (is.logical(data)) {
          dtype <- torch_bool()
        }
        
      }
      
      options <- torch_tensor_options(dtype = dtype, device = device, 
                                      pinned_memory = pin_memory)
      
      
      dimension <- dim(data)
      
      if (is.null(dimension)) {
        dimension <- length(data)
      }
      
      
      self$ptr <- cpp_torch_tensor(data, rev(dimension), options$ptr, 
                                   requires_grad)
    },
    print = function() {
      cat(sprintf("torch_tensor \n"))
      cpp_torch_tensor_print(self$ptr)
      invisible(self)
    },
    dtype = function() {
      torch_dtype$new(ptr = cpp_torch_tensor_dtype(self$ptr))
    },
    device = function() {
      Device$new(ptr = cpp_tensor_device(self$ptr))
    },
    dim = function() {
      length(self$size())
    },
    size = function(dim) {
      x <- cpp_tensor_dim(self$ptr)
      
      if (missing(dim))
        return(x)
      
      x[dim + 1]
    },
    numel = function() {
      cpp_tensor_numel(self$ptr)
    }
  ),
  active = list(
    shape = function() {
      self$size()
    }
  )
)

#' @export
torch_tensor <- function(data, dtype = NULL, device = NULL, requires_grad = FALSE, 
                         pin_memory = FALSE) {
  Tensor$new(data, dtype, device, requires_grad, pin_memory)
}

#' @export
as_array <- function(x) {
  UseMethod("as_array", x)
}

#' @export
as_array.torch_tensor <- function(x) {
  
  # dequantize before converting
  if (x$is_quantized())
    x <- x$dequantize()
  
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

is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

is_undefined_tensor <- function(x) {
  # TODO: correctrly transform undefined to NULL and fix.
  out <- paste0(capture.output(print(x)), collapse = " ")
  grepl("undefined", out)
}