#' Creates a slice
#'
#' Creates a slice object that can be used when indexing torch tensors.
#'
#' @param start (integer) starting index.
#' @param end (integer) the last selected index.
#' @param step (integer) the step between indexes.
#'
#' @examples
#' x <- torch_randn(10)
#' x[slc(start = 1, end = 5, step = 2)]
#' @export
slc <- function(start, end, step = 1) {
  if (is.infinite(end) && end > 0) {
    end <- .Machine$integer.max
  }

  if (end == -1) {
    end <- .Machine$integer.max
  } else if (end < -1) {
    end <- end + 1
  }

  structure(
    list(
      start = ifelse(start > 0, start - 1, start),
      end = end,
      step = step
    ),
    class = "slice"
  )
}

#' @export
print.slice <- function(x, ...) {
  cat("<slice>\n")
}

.d <- list(
  `:` = function(x, y) {
    if (inherits(x, "slice")) {
      x$step <- y
      return(x)
    }

    slc(start = x, end = y)
  },
  N = .Machine$integer.max,
  newaxis = NULL,
  `..` = structure(list(), class = "fill")
)

tensor_slice <- function(tensor, ..., drop = TRUE) {
  Tensor$new(ptr = Tensor_slice(tensor$ptr, environment(), drop = drop, mask = .d))
}

#' @export
`[.torch_tensor` <- function(x, ..., drop = TRUE) {
  tensor_slice(x, ..., drop = drop)
}

tensor_slice_put_ <- function(tensor, ..., value) {
  if (length(value) > 1 && is.atomic(value)) {
    value <- torch_tensor(value)
  }

  Tensor_slice_put(tensor$ptr, environment(), value, mask = .d)
}

#' @export
`[<-.torch_tensor` <- function(x, ..., value) {
  tensor_slice_put_(x, ..., value = value)
  invisible(x)
}
