
#' Upsample module
#'
#' Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
#' The input data is assumed to be of the form minibatch x channels x optional depth x
#' optional height] x width. Hence, for spatial inputs, we expect a 4D Tensor and for
#' volumetric inputs, we expect a 5D Tensor.
#'
#' The algorithms available for upsampling are nearest neighbor and linear, bilinear,
#' bicubic and trilinear for 3D, 4D and 5D input Tensor, respectively.
#'
#' One can either give a scale_factor or the target output size to calculate the
#' output size. (You cannot give both, as it is ambiguous)
#'
#' @param size (int or `Tuple[int]` or `Tuple[int, int]` or `Tuple[int, int, int]`, optional):
#'   output spatial sizes
#' @param scale_factor (float or `Tuple[float]` or `Tuple[float, float]` or `Tuple[float, float, float]`, optional):
#'   multiplier for spatial size. Has to match input size if it is a tuple.
#' @param mode (str, optional): the upsampling algorithm: one of `'nearest'`,
#'   `'linear'`, `'bilinear'`, `'bicubic'` and `'trilinear'`.
#'   Default: `'nearest'`
#' @param align_corners (bool, optional): if `TRUE`, the corner pixels of the input
#'   and output tensors are aligned, and thus preserving the values at
#'   those pixels. This only has effect when `mode` is
#'   `'linear'`, `'bilinear'`, or `'trilinear'`. Default: `FALSE`
#'
#' @examples
#' input <- torch_arange(start = 1, end = 4, dtype = torch_float())$view(c(1, 1, 2, 2))
#' nn_upsample(scale_factor = c(2), mode = "nearest")(input)
#' nn_upsample(scale_factor = c(2, 2), mode = "nearest")(input)
#' @export
nn_upsample <- nn_module(
  "nn_upsample",
  initialize = function(size = NULL, scale_factor = NULL, mode = "nearest",
                        align_corners = NULL) {
    self$size <- size
    self$scale_factor <- scale_factor
    self$mode <- mode
    self$align_corners <- align_corners
  },
  forward = function(input) {
    nnf_interpolate(input, self$size, self$scale_factor, self$mode, self$align_corners,
      recompute_scale_factor = FALSE
    )
  }
)
