#' Creates the reduction objet
#'
#' @name torch_reduction
#' @rdname torch_reduction
#' @concept tensor-attributes
#'
NULL

#' @rdname torch_reduction
#' @concept tensor-attributes
#' @export
torch_reduction_sum <- function() cpp_torch_reduction_sum()

#' @concept tensor-attributes
#' @rdname torch_reduction
#' @export
torch_reduction_mean <- function() cpp_torch_reduction_mean()

#' @concept tensor-attributes
#' @rdname torch_reduction
#' @export
torch_reduction_none <- function() cpp_torch_reduction_none()

reduction_enum <- function(x = c("mean", "sum", "none")) {
  if (x == "mean") {
    torch_reduction_mean()
  } else if (x == "sum") {
    torch_reduction_sum()
  } else {
    torch_reduction_none()
  }
}
