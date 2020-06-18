#' Returns a bool indicating if CUDA is currently available.
#'
#' @export
cuda_is_available <- function() {
  cpp_cuda_is_available()
}

#' Returns the index of a currently selected device.
#'
#' @export
cuda_current_device <- function() {
  cpp_cuda_current_device()
}

#' Returns the number of GPUs available.
#'
#' @export
cuda_device_count <- function() {
  cpp_cuda_device_count()
}