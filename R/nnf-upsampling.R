#' Interpolate
#'
#' Down/up samples the input to either the given `size` or the given
#' `scale_factor`
#'
#' The algorithm used for interpolation is determined by `mode`.
#'
#' Currently temporal, spatial and volumetric sampling are supported, i.e.
#' expected inputs are 3-D, 4-D or 5-D in shape.
#'
#' The input dimensions are interpreted in the form:
#' `mini-batch x channels x [optional depth] x [optional height] x width`.
#'
#' The modes available for resizing are: `nearest`, `linear` (3D-only),
#' `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`
#'
#' @param input (Tensor) the input tensor
#' @param size (int or `Tuple[int]` or `Tuple[int, int]` or `Tuple[int, int, int]`)
#'   output spatial size.
#' @param scale_factor (float or `Tuple[float]`) multiplier for spatial size.
#'   Has to match input size if it is a tuple.
#' @param mode (str) algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear'
#'  | 'bicubic' | 'trilinear' | 'area' Default: 'nearest'
#' @param align_corners (bool, optional) Geometrically, we consider the pixels
#'   of the input and output as squares rather than points. If set to TRUE,
#'   the input and output tensors are aligned by the center points of their corner
#'   pixels, preserving the values at the corner pixels. If set to False, the
#'   input and output tensors are aligned by the corner points of their corner pixels,
#'   and the interpolation uses edge value padding for out-of-boundary values,
#'   making this operation *independent* of input size when `scale_factor` is kept
#'   the same. This only has an effect when `mode`  is ``'linear'``, ``'bilinear'``,
#'   ``'bicubic'`` or ``'trilinear'``.  Default: ``False``
#' @param recompute_scale_factor (bool, optional) recompute the scale_factor
#'   for use in the interpolation calculation.  When `scale_factor` is passed
#'   as a parameter, it is used to compute the `output_size`.  If `recompute_scale_factor`
#'   is ```True`` or not specified, a new `scale_factor` will be computed based on
#'   the output and input sizes for use in the interpolation computation (i.e. the
#'   computation will be identical to if the computed `output_size` were passed-in
#'   explicitly).  Otherwise, the passed-in `scale_factor` will be used in the
#'   interpolation computation.  Note that when `scale_factor` is floating-point,
#'   the recomputed scale_factor may differ from the one passed in due to rounding
#'   and precision issues.
#'
#' @name nnf_interpolate
#'
#' @export
nnf_interpolate <- function(input, size = NULL, scale_factor = NULL,
                            mode = "nearest", align_corners = FALSE,
                            recompute_scale_factor = NULL) {
  dim <- input$dim() - 2

  if (is.null(align_corners)) align_corners <- FALSE

  # Process size and scale_factor.  Validate that exactly one is set.
  # Validate its length if it is a list, or expand it if it is a scalar.
  # After this block, exactly one of output_size and scale_factors will
  # be non-None, and it will be a list (or tuple).

  if (!is.null(size) && !is.null(scale_factor)) {
    value_error("only one of size or scale_factor should be defined")
  } else if (!is.null(size)) {
    scale_factors <- NULL
    if (length(size) == 1) {
      output_size <- rep(size, dim)
    } else {
      if (length(size) != dim) {
        value_error("size shape must match input shape. Input is {dim}D, size is {length(size)}")
      }
      output_size <- size
    }
  } else if (!is.null(scale_factor)) {
    output_size <- NULL
    if (length(scale_factor) == 1) {
      scale_factors <- rep(scale_factor, dim)
    } else {
      if (length(scale_factor) != dim) {
        value_error("scale_factor shape must match input shape. Input is {dim}D, size is {length(size)}")
      }
      scale_factors <- scale_factor
    }
  } else {
    value_error("either size or scale_factor should be defined")
  }

  # "area" mode always requires an explicit size rather than scale factor.
  # Re-use the recompute_scale_factor code path.
  if (mode == "area" && is.null(output_size)) {
    recompute_scale_factor <- TRUE
  }

  if (!is.null(recompute_scale_factor) && recompute_scale_factor) {
    # We compute output_size here, then un-set scale_factors.
    # The C++ code will recompute it based on the (integer) output size.
    output_size <- lapply(seq_len(dim), function(i) {
      as.integer(floor(input$size(i + 1)) * scale_factors[i])
    })
    scale_factors <- NULL
  }

  if (input$dim() == 3 && mode == "nearest") {
    return(
      torch_upsample_nearest1d(
        input = input, output_size = output_size,
        scale_factors = scale_factors
      )
    )
  }

  if (input$dim() == 4 && mode == "nearest") {
    return(torch_upsample_nearest2d(
      input = input, output_size = output_size,
      scale_factors = scale_factors
    ))
  }

  if (input$dim() == 5 && mode == "nearest") {
    return(torch_upsample_nearest3d(
      input = input, output_size = output_size,
      scale_factors = scale_factors
    ))
  }

  if (input$dim() == 3 && mode == "area") {
    return(torch_adaptive_avg_pool1d(input, output_size = output_size))
  }

  if (input$dim() == 4 && mode == "area") {
    return(torch_adaptive_avg_pool2d(input, output_size = output_size))
  }

  if (input$dim() == 5 && mode == "area") {
    return(torch_adaptive_avg_pool3d(input, output_size = output_size))
  }

  if (input$dim() == 3 && mode == "linear") {
    return(torch_upsample_linear1d(
      input = input, output_size = output_size,
      align_corners = align_corners,
      scale_factors = scale_factors
    ))
  }

  if (input$dim() == 3 && mode == "bilinear") {
    not_implemented_error("Got 3D input, but bilinear mode needs 4D input")
  }

  if (input$dim() == 3 && mode == "trilinear") {
    not_implemented_error("Got 3D input, but trilinear mode needs 5D input")
  }

  if (input$dim() == 4 && mode == "linear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 3D input")
  }

  if (input$dim() == 4 && mode == "bilinear") {
    return(torch_upsample_bilinear2d(
      input = input, output_size = output_size,
      align_corners = align_corners,
      scale_factors = scale_factors
    ))
  }

  if (input$dim() == 4 && mode == "trilinear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 5D input")
  }

  if (input$dim() == 5 && mode == "linear") {
    not_implemented_error("Got 5D input, but trilinear mode needs 3D input")
  }

  if (input$dim() == 5 && mode == "bilinear") {
    not_implemented_error("Got 5D input, but bilinear mode needs 4D input")
  }

  if (input$dim() == 5 && mode == "trilinear") {
    return(torch_upsample_trilinear3d(
      input = input, output_size = output_size,
      align_corners = align_corners,
      scale_factors = scale_factors
    ))
  }

  if (input$dim() == 4 && mode == "bicubic") {
    return(torch_upsample_bicubic2d(
      input = input, output_size = output_size,
      align_corners = align_corners,
      scale_factors = scale_factors
    ))
  }

  not_implemented_error(
    "Input Error: Only 3D, 4D and 5D input Tensors supported",
    " (got {input$dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear",
    " (got {mode})"
  )
}
