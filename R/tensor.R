#' @include R7.R

Tensor <- R7Class(
  classname = "torch_tensor", 
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
      
      x[dim]
    },
    numel = function() {
      cpp_tensor_numel(self$ptr)
    },
    to = function(dtype = NULL, device = NULL, other = NULL, non_blocking = FALSE, 
                  copy = FALSE, memory_format = torch_preserve_format()) {
      
    
      if (!is.null(other))
        args <- list(other = other)
      else if (is.null(device))
        args <- list(dtype = dtype)
      else
        args <- list(dtype = dtype, device = device) 
      
      args$non_blocking <- non_blocking
      args$copy <- copy
      args$memory_format <- memory_format
      
      if (is.null(args$dtype) && is.null(args$other))
        args$dtype <- self$dtype()
      
      do.call(private$`_to`, args)
    },
    cuda = function(device=NULL, non_blocking=FALSE, memory_format=torch_preserve_format()) {
      
      if (is.null(device))
        device <- torch_device("cuda")
        
      if (!device$type == "cuda")
        value_error("You must pass a cuda device.")
      
      self$to(device = device, non_blocking=non_blocking, memory_format = memory_format)
    },
    cpu = function(memory_format=torch_preserve_format()) {
      self$to(device = torch_device("cpu"), memory_format = memory_format)
    },
    stride = function(dim) {
      if (missing(dim)) {
        d <- self$dim()
        sapply(seq_len(d), private$`_stride`)
      } else {
        private$`_stride`(dim)
      }
    },
    is_contiguous = function() {
      cpp_tensor_is_contiguous(self$ptr)
    },
    copy_ = function(src, non_blocking = FALSE) {
      
      if (is_null_external_pointer(self$ptr))
        self$ptr <- torch_empty_like(src)$ptr
      
      private$`_copy_`(src, non_blocking)
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
  
  if (x$device()$type == "cuda")
    runtime_error("Can't convert cuda tensor to R. Convert to cpu tensor before.")
  
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
  cpp_tensor_is_undefined(x$ptr)
}

