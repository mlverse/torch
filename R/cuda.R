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

#' Returns the major and minor CUDA capability of `device`
#'
#' @param device Integer value of the CUDA device to return capabilities of.
#'
#' @export
cuda_get_device_capability <- function(device = cuda_current_device()) {
  if (device < 0 | device >= cuda_device_count()) {
    stop(paste("device must be an integer between 0 and the number of devices minus 1"))
  }
  res <- as.integer(cpp_cuda_get_device_capability(device))
  names(res) <- c("Major", "Minor")
  res
}
