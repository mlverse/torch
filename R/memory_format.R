MemoryFormat <- R7Class(
  classname = "torch_memory_format",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      ptr
    },
    print = function() {
      cat("torch_", cpp_memory_format_to_string(ptr), "_format", sep = "")
    }
  )
)

#' Memory format
#'
#' Returns the correspondent memory format.
#'
#' @name torch_memory_format
#' @rdname torch_memory_format
NULL

#' @rdname torch_memory_format
#' @export
torch_contiguous_format <- function() {
  MemoryFormat$new(cpp_torch_contiguous_format())
}

#' @rdname torch_memory_format
#' @export
torch_preserve_format <- function() {
  MemoryFormat$new(cpp_torch_preserve_format())
}

#' @rdname torch_memory_format
#' @export
torch_channels_last_format <- function() {
  MemoryFormat$new(cpp_torch_channels_last_format())
}

#' @export
`==.torch_memory_format` <- function(e1, e2) {
  cpp_memory_format_to_string(e1) == cpp_memory_format_to_string(e2)
}

#' Check if an object is a memory format
#'
#' @param x object to check
#'
#' @export
is_torch_memory_format <- function(x) {
  inherits(x, "torch_memory_format")
}
