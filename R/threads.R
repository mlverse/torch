#' Number of threads
#'
#' @description
#' Get and set the numbers used by torch computations.
#'
#' @param num_threads number of threads to set.
#'
#' @name threads
#'
#' @details
#' For details see the [CPU threading article](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html?highlight=set_num_threads)
#' in the PyTorch documentation.
#'
#' @note torch_set_threads do not work on macOS system as it must be 1.
#'
#' @rdname threads
NULL

#' @rdname threads
#' @export
torch_set_num_threads <- function(num_threads) {
  cpp_set_num_threads(num_threads)
}

#' @rdname threads
#' @export
torch_set_num_interop_threads <- function(num_threads) {
  cpp_set_num_interop_threads(num_threads)
}

#' @rdname threads
#' @export
torch_get_num_interop_threads <- function() {
  cpp_get_num_interop_threads()
}

#' @rdname threads
#' @export
torch_get_num_threads <- function() {
  cpp_get_num_threads()
}
