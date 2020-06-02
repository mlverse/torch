#' Pixel_shuffle
#'
#' Rearranges elements in a tensor of shape \eqn{(*, C \times r^2, H, W)} to a
#' tensor of shape \eqn{(*, C, H \times r, W \times r)}.
#'
#' @param input (Tensor) the input tensor
#' @param upscale_factor (int) factor to increase spatial resolution by
#'
#' @export
nnf_pixel_shuffle <- function(input, upscale_factor) {
  torch_pixel_shuffle(input, upscale_factor)
}
