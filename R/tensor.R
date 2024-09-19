#' @include R7.R

Tensor <- R7Class(
  classname = "torch_tensor",
  public = list(
    initialize = function(data = NULL, dtype = NULL, device = NULL, requires_grad = FALSE,
                          pin_memory = FALSE, ptr = NULL) {
      if (!is.null(ptr)) {
        return(ptr)
      }

      torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory)
    },
    print = function(n = 30) {
      cat("torch_tensor\n")
      if (torch_is_complex(self) && !is_meta_device(self$device)) {
        dtype <- as.character(self$dtype)
        device <- toupper(self$device$type)
        shape <- paste(self$shape, collapse = ",")
        cat(cli::cli_format_method(
          cli::cli_alert_info("Use {.var $real} or {.var $imag} to print the contents of this tensor.")
        ))
        cat("\n")
        cat("[ ", device, dtype, "Type{", shape, "} ]", sep = "")
      } else if (is_undefined_tensor(self) || !is_meta_device(self$device)) {
        cpp_torch_tensor_print(self$ptr, n)
      } else {
        cat("...\n")
        dtype <- as.character(self$dtype)
        shape <- paste(self$shape, collapse = ",")
        cat("[ META", dtype, "Type{", shape, "} ]", sep = "")
      }
      if (!is_undefined_tensor(self)) {
        if (!is.null(self$ptr$grad_fn)) {
          cat("[ grad_fn = <")
          cat(self$ptr$grad_fn$print(newline = FALSE))
          cat("> ]")
        } else if (self$ptr$requires_grad && is.null(self$ptr$grad_fn)) {
          cat("[ requires_grad = TRUE ]")
        }
      }
      cat("\n")
      invisible(self)
    },
    dim = function() {
      cpp_tensor_ndim(self)
    },
    length = function() {
      prod(dim(self))
    },
    size = function(dim) {
      x <- cpp_tensor_dim(self$ptr)

      if (missing(dim)) {
        return(x)
      }

      if (dim == 0) runtime_error("Indexing starts at 1 and got a 0.")

      if (dim > 0) {
        x[dim]
      } else {
        rev(x)[abs(dim)]
      }
    },
    element_size = function() {
      cpp_tensor_element_size(self$ptr)
    },
    numel = function() {
      cpp_tensor_numel(self$ptr)
    },
    to = function(dtype = NULL, device = NULL, other = NULL, non_blocking = FALSE,
                  copy = FALSE, memory_format = NULL) {

      has_device <- !is.null(device)
      has_dtype <- !is.null(dtype)
      has_other <- !is.null(other)

      if (has_other) {
        # can't have device and dtype
        if (has_device || has_dtype) {
          cli::cli_abort("Had {.arg other} but {.arg device} or {.arg dtype} are non {.val NULL}")
        }

        return(private$`_to`(other = other, non_blocking = non_blocking, copy = copy))
      }

      if (!has_dtype) {
        dtype <- self$dtype
      }

      if (has_device) {
        private$`_to`(
          dtype = dtype,
          device = device,
          non_blocking = non_blocking,
          copy = copy,
          memory_format = memory_format
        )
      } else {
        private$`_to`(
          dtype = dtype,
          non_blocking = non_blocking,
          copy = copy,
          memory_format = memory_format
        )
      }
    },
    bool = function(memory_format = torch_preserve_format()) {
      self$to(torch_bool(), memory_format = memory_format)
    },
    cuda = function(device = NULL, non_blocking = FALSE, memory_format = torch_preserve_format()) {
      if (is.null(device)) {
        device <- torch_device("cuda")
      }

      if (!device$type == "cuda") {
        value_error("You must pass a cuda device.")
      }

      self$to(device = device, non_blocking = non_blocking, memory_format = memory_format)
    },
    cpu = function(memory_format = torch_preserve_format()) {
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
      if (is_null_external_pointer(self$ptr)) {
        g <- torch_empty_like(src, requires_grad = src$requires_grad)
        # this is the only way modify `self` in place.
        # changing it's address in the C side and
        # adding a protection to `g` so it only
        # gets destroyed when `self` itself is destroyed.
        set_xptr_address(self, g)
        set_xptr_protected(self, g)
      }

      self$private$`_copy_`(src, non_blocking)
    },
    topk = function(k, dim = -1L, largest = TRUE, sorted = TRUE) {
      o <- private$`_topk`(k, dim, largest, sorted)
      o[[2]]$add_(1L)
      o
    },
    scatter = function(dim, index, src) {
      if (is_torch_tensor(src)) {
        private$`_scatter`(dim, index, src = src)
      } else {
        private$`_scatter`(dim, index, value = src)
      }
    },
    scatter_ = function(dim, index, src) {
      if (is_torch_tensor(src)) {
        private$`_scatter_`(dim, index, src = src)
      } else {
        private$`_scatter_`(dim, index, value = src)
      }
    },
    has_names = function() {
      cpp_tensor_has_names(self$ptr)
    },
    rename = function(...) {
      nms <- prep_names(..., self = self)
      private$`_rename`(nms)
    },
    rename_ = function(...) {
      nms <- prep_names(..., self = self)
      private$`_rename_`(nms)
    },
    narrow = function(dim, start, length) {
      start <- torch_scalar_tensor(start, dtype = torch_int64())
      if (start$item() == 0) {
        value_error("start indexing starts at 1")
      }
      start <- start - 1L
      private$`_narrow`(dim, start, length)
    },
    narrow_copy = function(dim, start, length) {
      if (start == 0) {
        value_error("start indexing starts at 1")
      }
      start <- start - 1L
      private$`_narrow_copy`(dim, start, length)
    },
    max = function(dim, other, keepdim = FALSE) {
      if (missing(dim) && missing(other)) {
        return(private$`_max`())
      }

      if (!missing(dim) && !missing(other)) {
        value_error("Can't set other and dim arguments.")
      }

      if (missing(dim)) {
        return(private$`_max`(other = other))
      }

      # dim is not missing
      o <- private$`_max`(dim = dim, keepdim = keepdim)
      o[[2]] <- o[[2]] + 1L # make 1 based
      o
    },
    min = function(dim, other, keepdim = FALSE) {
      if (missing(dim) && missing(other)) {
        return(private$`_min`())
      }

      if (!missing(dim) && !missing(other)) {
        value_error("Can't set other and dim arguments.")
      }

      if (missing(dim)) {
        return(private$`_min`(other = other))
      }

      # dim is not missing
      o <- private$`_min`(dim = dim, keepdim = keepdim)
      o[[2]] <- o[[2]] + 1L # make 1 based
      o
    },
    argsort = function(dim = -1L, descending = FALSE) {
      private$`_argsort`(dim = dim, descending = descending)$add_(1L, 1L)
    },
    argmax = function(dim = NULL, keepdim = FALSE) {
      o <- private$`_argmax`(dim = dim, keepdim = keepdim)
      o <- o$add_(1L, 1L)
      o
    },
    argmin = function(dim = NULL, keepdim = FALSE) {
      o <- private$`_argmin`(dim = dim, keepdim = keepdim)
      o <- o$add_(1L, 1L)
      o
    },
    sort = function(dim = -1L, descending = FALSE, stable) {
      if (missing(stable)) {
        o <- private$`_sort`(dim = dim, descending = descending)
      } else {
        o <- private$`_sort`(dim = dim, descending = descending, stable = stable)
      }
      o[[2]]$add_(1L)
      o
    },
    norm = function(p = 2, dim, keepdim = FALSE, dtype) {
      torch_norm(self, p, dim, keepdim, dtype)
    },
    split = function(split_size, dim = 1L) {
      if (length(split_size) > 1) {
        self$split_with_sizes(split_size, dim)
      } else {
        private$`_split`(split_size, dim)
      }
    },
    nonzero = function(as_list = FALSE) {
      if (!as_list) {
        return(private$`_nonzero`() + 1L)
      } else {
        o <- private$`_nonzero_numpy`()
        return(lapply(o, function(x) x + 1L))
      }
    },
    view = function(size) {
      private$`_view`(size = size)
    },
    bincount = function(weights = list(), minlength = 0L) {
      to_index_tensor(self)$private$`_bincount`(weights = weights, minlength = minlength)
    },
    is_sparse = function() {
      cpp_method_Tensor_is_sparse(self)
    },
    movedim = function(source, destination) {
      private$`_movedim`(as_1_based_dim(source), as_1_based_dim(destination))
    },
    float = function() {
      self$to(dtype=torch_float())
    },
    double = function() {
      self$to(dtype=torch_double())
    },
    half = function() {
      self$to(dtype=torch_half())
    },
    clone = function(...) {
      x <- private$`_clone`(...)
      attributes(x) <- attributes(self)

      return(x)
    },
    detach = function(...) {
      x <- private$`_detach`(...)
      attributes(x) <- attributes(self)

      return(x)
    }
  ),
  active = list(
    shape = function() {
      self$size()
    },
    dtype = function() {
      torch_dtype$new(ptr = cpp_torch_tensor_dtype(self$ptr))
    },
    device = function() {
      Device$new(ptr = cpp_tensor_device(self$ptr))
    },
    is_cuda = function() {
      self$device$type == "cuda"
    },
    ndim = function() {
      self$dim()
    },
    names = function() {
      if (!self$has_names()) {
        return(NULL)
      }

      p <- cpp_tensor_names(self$ptr)
      DimnameList$new(ptr = p)$to_r()
    },
    is_leaf = function() {
      private$`_is_leaf`()
    },
    real = function(x) {
      if (missing(x)) return(torch_real(self))
      self$real$copy_(x)
    },
    imag = function(x) {
      if (missing(x)) return(torch_imag(self))
      self$imag$copy_(x)
    }
  )
)

prep_names <- function(..., self) {
  new_nms <- unlist(rlang::list2(...))

  if (rlang::is_named(new_nms)) {
    if (!self$has_names()) {
      runtime_error("The tensor doesn't have names so you can't rename a dimension.")
    }

    nms <- self$names
    nms[which(nms == names(new_nms))] <- new_nms
  } else {
    nms <- new_nms
  }
  nms
}

#' Converts R objects to a torch tensor
#'
#' @param data an R atomic vector, matrix or array
#' @param dtype a [torch_dtype] instance
#' @param device a device creted with [torch_device()]
#' @param requires_grad if autograd should record operations on the returned tensor.
#' @param pin_memory If set, returned tensor would be allocated in the pinned memory.
#'
#' @examples
#' torch_tensor(c(1, 2, 3, 4))
#' torch_tensor(c(1, 2, 3, 4), dtype = torch_int())
#' @export
torch_tensor <- function(data, dtype = NULL, device = NULL, requires_grad = FALSE,
                         pin_memory = FALSE) {
  Tensor$new(data, dtype, device, requires_grad, pin_memory)
}

#' Converts to array
#'
#' @param x object to be converted into an array
#'
#' @export
as_array <- function(x) {
  UseMethod("as_array", x)
}

#' @export
as.array.torch_tensor <- function(x, ...) {
  as_array(x)
}

#' @export
as.matrix.torch_tensor <- function(x, ...) {
  as.matrix(as_array(x))
}

as_array_impl <- function(x) {
  # move tensor to cpu before copying to R
  x <- x$cpu()

  if (x$is_complex()) {
    out <- complex(
      real = as.array(x$real),
      imaginary = as.array(x$imag)
    )
    dim(out) <- dim(x)
    return(out)
  }

  a <- cpp_as_array(x)

  if (length(a$dim) <= 1L) {
    out <- a$vec
  } else if (length(a$dim) == 2L) {
    out <- t(matrix(a$vec, ncol = a$dim[1], nrow = a$dim[2]))
  } else {
    out <- aperm(array(a$vec, dim = rev(a$dim)), seq(length(a$dim), 1))
  }

  if (x$dtype == torch_long() && !inherits(out, "integer64")) {
    class(out) <- c(class(out), "integer64")
  }

  out
}

#' @export
as_array.torch_tensor <- function(x) {
  # dequantize before converting
  if (x$is_quantized()) {
    x <- x$dequantize()
  }

  # auto convert to int32 if long.
  if (x$dtype == torch_long()) {
    if ((x > .Machine$integer.max)$any()$item()) {
      warn("Converting integers > .Machine$integer.max is undefined and returns wrong results. Use as.integer64(x)")
    }

    x <- x$to(dtype = torch_int32())
  }


  out <- as_array_impl(x)

  out
}

is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

#' Checks if a tensor is undefined
#'
#' @param x tensor to check
#'
#' @export
is_undefined_tensor <- function(x) {
  cpp_tensor_is_undefined(x$ptr)
}

#' @importFrom bit64 as.integer64
#' @export
as.integer64.torch_tensor <- function(x, keep.names = FALSE, ...) {
  x <- x$to(dtype = torch_long())
  as_array_impl(x)
}

make_str_torch_tensor <- function(object) {
  dtype <- object$dtype$.type()

  dims <- dim(object)
  dims <- paste(paste0("1:", dims), collapse = ", ")

  out <- paste0(dtype, " [", dims, "]")
  out
}

#' @export
str.torch_tensor <- function(object, ...) {
  out <- make_str_torch_tensor(object)
  cat(out)
  cat("\n")
}

Tensor$set("active", "ptr", function() {
  self
})

tensor_to_complex <- function(x) {
  torch_complex(
    torch_tensor(Re(x), dtype = torch_double()),
    torch_tensor(Im(x), dtype = torch_double())
  )
}

#' Creates a tensor from a buffer of memory
#'
#' It creates a tensor without taking ownership of the memory it points to.
#' You must call `clone` if you want to copy the memory over a new tensor.
#'
#' @param buffer An R atomic object containing the data in a contiguous array.
#' @param tensor Tensor object that will be converted into a buffer.
#' @param shape The shape of the resulting tensor.
#' @param dtype A torch data type for the tresulting tensor.
#'
#' @export
torch_tensor_from_buffer <- function(buffer, shape, dtype = "float") {
  cpp_tensor_from_buffer(buffer, shape, list(dtype=dtype))
}

#' @describeIn torch_tensor_from_buffer Creates a raw vector containing the tensor data. Causes a data copy.
#' @export
buffer_from_torch_tensor <- function(tensor) {
  cpp_buffer_from_tensor(tensor)
}
