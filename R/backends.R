
#' CuDNN is available
#'
#' @export
backends_cudnn_is_available <- function() {
  cpp_cudnn_is_available()
}

backends_cudnn_enabled <- backends_cudnn_is_available

#' CuDNN version
#'
#' @export
backends_cudnn_version <- function() {
  if (!backends_cudnn_is_available()) {
    rlang::abort("CuDNN is not available.")
  }

  v <- cpp_cudnn_runtime_version()
  major <- trunc(v / 1000)
  minor <- trunc((v - major * 1000) / 100)
  patch <- v - major * 1000 - minor * 100
  numeric_version(paste(major, minor, patch, sep = "."))
}

#' MKLDNN is available
#' @return Returns whether LibTorch is built with MKL-DNN support.
#' @export
backends_mkldnn_is_available <- function() {
  cpp_backends_mkldnn_is_available()
}

#' MKL is available
#' @return Returns whether LibTorch is built with MKL support.
#' @export
backends_mkl_is_available <- function() {
  cpp_backends_mkl_is_available()
}

#' OpenMP is available
#' @return Returns whether LibTorch is built with OpenMP support.
#' @export
backends_openmp_is_available <- function() {
  cpp_backends_openmp_is_available()
}
