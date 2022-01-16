#' Fold
#'
#' Combines an array of sliding local blocks into a large containing
#' tensor.
#'
#' @section Warning:
#'
#' Currently, only 4-D output tensors (batched image-like tensors) are
#' supported.
#'
#' @inheritParams nnf_unfold
#' @param output_size the shape of the spatial dimensions of the output (i.e.,
#'   `output$sizes()[-c(1,2)]`)
#'
#' @export
nnf_fold <- function(input, output_size, kernel_size, dilation = 1, padding = 0, stride = 1) {
  torch_col2im(
    self = input, output_size = nn_util_pair(output_size),
    kernel_size = nn_util_pair(kernel_size), dilation = nn_util_pair(dilation),
    padding = nn_util_pair(padding), stride = nn_util_pair(stride)
  )
}

#' Unfold
#'
#' Extracts sliding local blocks from an batched input tensor.
#'
#' @section Warning:
#'
#' Currently, only 4-D input tensors (batched image-like tensors) are
#' supported.
#'
#' @section Warning:
#'
#' More than one element of the unfolded tensor may refer to a single
#' memory location. As a result, in-place operations (especially ones that
#' are vectorized) may result in incorrect behavior. If you need to write
#' to the tensor, please clone it first.
#'
#' @param input the input tensor
#' @param kernel_size the size of the sliding blocks
#' @param dilation a parameter that controls the stride of elements within the
#'   neighborhood. Default: 1
#' @param padding implicit zero padding to be added on both sides of input.
#'   Default: 0
#' @param stride the stride of the sliding blocks in the input spatial dimensions.
#'   Default: 1
#'
#' @export
nnf_unfold <- function(input, kernel_size, dilation = 1, padding = 0, stride = 1) {
  if (input$dim() == 4) {
    torch_im2col(
      input, nn_util_pair(kernel_size), nn_util_pair(dilation),
      nn_util_pair(padding), nn_util_pair(stride)
    )
  } else {
    not_implemented_error("Input Error: Only 4D input Tensors are supported (got {input$dim()}D)")
  }
}
