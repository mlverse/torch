#' Waits for all kernels in all streams on the MPS device to complete.
#'
#' @export
backends_mps_synchronize <- function() {
  cpp_mps_synchronize()
}
