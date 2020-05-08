#' @export
torch_reduction_sum <- function() cpp_torch_reduction_sum()

#' @export
torch_reduction_mean <- function() cpp_torch_reduction_mean()

#' @export
torch_reduction_none <- function() cpp_torch_reduction_none()

reduction_enum <- function(x = c("mean", "sum", "none")) {
  if (x == "mean")
    torch_reduction_mean()
  else if (x == "sum")
    torch_reduction_sum()
  else 
    torch_reduction_none()
}