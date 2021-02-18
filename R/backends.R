backends_cudnn_enabled <- function() {
  TRUE
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
