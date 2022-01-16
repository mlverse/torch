#' Integer type info
#'
#' A list that represents the numerical properties of a integer
#' type.
#'
#' @param dtype dtype to get information from.
#' @concept tensor-attributes
#'
#' @export
torch_iinfo <- function(dtype) {
  if (dtype == torch_int32()) {
    list(
      bits = 32,
      max = bit64::as.integer64("2147483647"),
      min = bit64::as.integer64("-2147483648")
    )
  } else if (dtype == torch_int64()) {
    list(
      bits = 64,
      max = bit64::as.integer64("9223372036854775807"),
      min = bit64::as.integer64("-9223372036854775808")
    )
  } else if (dtype == torch_int16()) {
    list(
      bits = 16,
      max = 32767L,
      min = -32768L
    )
  } else {
    value_error("dtype must be an integer type.")
  }
}

#' Floating point type info
#'
#' A list that represents the numerical properties of a
#' floating point torch.dtype
#'
#' @param dtype dtype to check information
#' @concept tensor-attributes
#'
#' @export
torch_finfo <- function(dtype) {
  if (dtype == torch_float32()) {
    list(
      bits = 32,
      max = 3.4028234663852886e+38,
      min = -3.4028234663852886e+38,
      eps = 1.1920928955078125e-07,
      tiny = 1.1754943508222875e-38
    )
  } else if (dtype == torch_float64()) {
    list(
      bits = 64,
      max = 1.7976931348623157e+308,
      min = -1.7976931348623157e+308,
      eps = 2.220446049250313e-16,
      tiny = 2.2250738585072014e-308
    )
  } else if (dtype == torch_float16()) {
    list(
      bits = 16,
      max = 65504.0,
      min = -65504.0,
      eps = 0.0009765625,
      tiny = 6.103515625e-05
    )
  } else {
    value_error("dtype must be a float type.")
  }
}
