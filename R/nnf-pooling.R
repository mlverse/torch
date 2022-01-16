#' Avg_pool1d
#'
#' Applies a 1D average pooling over an input signal composed of several
#' input planes.
#'
#'
#' @param input input tensor of shape (minibatch , in_channels , iW)
#' @param kernel_size the size of the window. Can be a single number or a
#'  tuple `(kW,)`.
#' @param stride the stride of the window. Can be a single number or a tuple
#'  `(sW,)`. Default: `kernel_size`
#' @param padding implicit zero paddings on both sides of the input. Can be a
#'   single number or a tuple `(padW,)`. Default: 0
#' @param ceil_mode when True, will use `ceil` instead of `floor` to compute the
#'   output shape. Default: `FALSE`
#' @param count_include_pad when True, will include the zero-padding in the
#'   averaging calculation. Default: `TRUE`
#'
#' @export
nnf_avg_pool1d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                           count_include_pad = TRUE) {
  if (is.null(stride)) stride <- list()
  torch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
}

#' Avg_pool2d
#'
#' Applies 2D average-pooling operation in \eqn{kH * kW} regions by step size
#' \eqn{sH * sW} steps. The number of output features is equal to the number of
#' input planes.
#'
#'
#' @param input input tensor (minibatch, in_channels , iH , iW)
#' @param kernel_size size of the pooling region. Can be a single number or a
#'   tuple `(kH, kW)`
#' @param stride stride of the pooling operation. Can be a single number or a
#'   tuple `(sH, sW)`. Default: `kernel_size`
#' @param padding implicit zero paddings on both sides of the input. Can be a
#'   single number or a tuple `(padH, padW)`. Default: 0
#' @param ceil_mode when True, will use `ceil` instead of `floor` in the formula
#'   to compute the output shape. Default: `FALSE`
#' @param count_include_pad when True, will include the zero-padding in the
#'   averaging calculation. Default: `TRUE`
#' @param divisor_override if specified, it will be used as divisor, otherwise
#'   size of the pooling region will be used. Default: `NULL`
#'
#' @export
nnf_avg_pool2d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                           count_include_pad = TRUE, divisor_override = NULL) {
  if (is.null(stride)) stride <- list()
  torch_avg_pool2d(
    input, kernel_size, stride, padding, ceil_mode, count_include_pad,
    divisor_override
  )
}

#' Avg_pool3d
#'
#' Applies 3D average-pooling operation in \eqn{kT * kH * kW} regions by step
#' size \eqn{sT * sH * sW} steps. The number of output features is equal to
#' \eqn{\lfloor \frac{ \mbox{input planes} }{sT} \rfloor}.
#'
#' @param input input tensor (minibatch, in_channels , iT * iH , iW)
#' @param kernel_size size of the pooling region. Can be a single number or a
#'   tuple `(kT, kH, kW)`
#' @param stride stride of the pooling operation. Can be a single number or a
#'   tuple `(sT, sH, sW)`. Default: `kernel_size`
#' @param padding implicit zero paddings on both sides of the input. Can be a
#'   single number or a tuple `(padT, padH, padW)`, Default: 0
#' @param ceil_mode when True, will use `ceil` instead of `floor` in the formula
#'   to compute the output shape
#' @param count_include_pad when True, will include the zero-padding in the
#'   averaging calculation
#' @param divisor_override NA if specified, it will be used as divisor, otherwise
#'   size of the pooling region will be used. Default: `NULL`
#'
#' @export
nnf_avg_pool3d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                           count_include_pad = TRUE, divisor_override = NULL) {
  if (is.null(stride)) stride <- list()
  torch_avg_pool3d(
    input, kernel_size, stride, padding, ceil_mode, count_include_pad,
    divisor_override
  )
}

#' Max_pool1d
#'
#' Applies a 1D max pooling over an input signal composed of several input
#' planes.
#'
#' @inheritParams nnf_avg_pool1d
#' @param dilation controls the spacing between the kernel points; also known as
#'   the à trous algorithm.
#' @param return_indices whether to return the indices where the max occurs.
#'
#' @export
nnf_max_pool1d <- function(input, kernel_size, stride = NULL, padding = 0, dilation = 1,
                           ceil_mode = FALSE, return_indices = FALSE) {
  if (is.null(stride)) stride <- list()

  if (return_indices) {
    torch_max_pool1d_with_indices(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  } else {
    torch_max_pool1d(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  }
}

#' Max_pool2d
#'
#' Applies a 2D max pooling over an input signal composed of several input
#' planes.
#'
#' @inheritParams nnf_avg_pool2d
#' @inheritParams nnf_max_pool1d
#'
#'
#' @export
nnf_max_pool2d <- function(input, kernel_size, stride = kernel_size, padding = 0, dilation = 1,
                           ceil_mode = FALSE, return_indices = FALSE) {
  if (return_indices) {
    torch_max_pool2d_with_indices(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  } else {
    torch_max_pool2d(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  }
}

#' Max_pool3d
#'
#' Applies a 3D max pooling over an input signal composed of several input
#' planes.
#'
#' @inheritParams nnf_avg_pool3d
#' @inheritParams nnf_max_pool1d
#'
#' @export
nnf_max_pool3d <- function(input, kernel_size, stride = NULL, padding = 0, dilation = 1,
                           ceil_mode = FALSE, return_indices = FALSE) {
  if (return_indices) {
    torch_max_pool3d_with_indices(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  } else {
    torch_max_pool3d(
      input, kernel_size, stride, padding, dilation,
      ceil_mode
    )
  }
}

#' Adaptive_max_pool1d
#'
#' Applies a 1D adaptive max pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool1d
#' @param output_size the target output size (single integer)
#' @param return_indices whether to return pooling indices. Default: `FALSE`
#'
#' @export
nnf_adaptive_max_pool1d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool1d(input, output_size)
  if (!return_indices) {
    o <- o[[1]]
  }

  o
}

#' Adaptive_max_pool2d
#'
#' Applies a 2D adaptive max pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool2d
#' @param output_size the target output size (single integer or double-integer tuple)
#' @param return_indices whether to return pooling indices. Default: `FALSE`
#'
#'
#' @export
nnf_adaptive_max_pool2d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool2d(input, output_size)
  if (!return_indices) {
    o <- o[[1]]
  }

  o
}

#' Adaptive_max_pool3d
#'
#' Applies a 3D adaptive max pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool3d
#' @param output_size the target output size (single integer or triple-integer tuple)
#' @param return_indices whether to return pooling indices. Default:`FALSE`
#'
#' @export
nnf_adaptive_max_pool3d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool3d(input, output_size)
  if (!return_indices) {
    o <- o[[1]]
  }

  o
}

#' Adaptive_avg_pool1d
#'
#' Applies a 1D adaptive average pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool1d
#' @param output_size the target output size (single integer)
#'
#' @export
nnf_adaptive_avg_pool1d <- function(input, output_size) {
  torch_adaptive_avg_pool1d(input, output_size)
}

#' Adaptive_avg_pool2d
#'
#' Applies a 2D adaptive average pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool2d
#' @param output_size the target output size (single integer or double-integer tuple)
#'
#' @export
nnf_adaptive_avg_pool2d <- function(input, output_size) {
  torch_adaptive_avg_pool2d(input, output_size)
}

#' Adaptive_avg_pool3d
#'
#' Applies a 3D adaptive average pooling over an input signal composed of
#' several input planes.
#'
#' @inheritParams nnf_avg_pool3d
#' @param output_size the target output size (single integer or triple-integer tuple)
#'
#' @export
nnf_adaptive_avg_pool3d <- function(input, output_size) {
  torch_adaptive_avg_pool3d(input, output_size)
}

unpool_output_size <- function(input, kernel_size, stride, padding, output_size) {
  input_size <- input$size()
  default_size <- list()
  for (d in seq_along(kernel_size)) {
    default_size[[d]] <- (input_size[d + 2] - 1) * stride[d] + kernel_size[d] -
      2 * padding[d]
  }

  if (is.null(output_size)) {
    ret <- default_size
  } else {
    if (length(output_size) == (length(kernel_size) + 2)) {
      output_size <- output_size[-c(1, 2)]
    }

    if (length(output_size) != length(kernel_size)) {
      value_error(
        "output_size should be a sequence containing ",
        "{length(kernel_size)} or {length(kernel_size) + 2} elements",
        "but it has a length of '{length(output_size)}'"
      )
    }

    for (d in seq_along(kernel_size)) {
      min_size <- default_size[[d]] - stride[d]
      max_size <- default_size[[d]] + stride[d]
    }

    ret <- output_size
  }

  ret
}

#' Max_unpool1d
#'
#' Computes a partial inverse of `MaxPool1d`.
#'
#' @param input the input Tensor to invert
#' @param indices the indices given out by max pool
#' @param kernel_size Size of the max pooling window.
#' @param stride Stride of the max pooling window. It is set to kernel_size by default.
#' @param padding Padding that was added to the input
#' @param output_size the targeted output size
#'
#' @export
nnf_max_unpool1d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL) {
  if (is.null(stride) || length(stride) == 0) {
    stride <- kernel_size
  }

  output_size <- unpool_output_size(
    input, kernel_size, stride, padding,
    output_size
  )

  output_size <- c(output_size, 1)

  torch_max_unpool2d(input$unsqueeze(3), indices$unsqueeze(3), output_size)$squeeze(3)
}

#' Max_unpool2d
#'
#' Computes a partial inverse of `MaxPool2d`.
#'
#' @inheritParams nnf_max_unpool1d
#'
#' @export
nnf_max_unpool2d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL) {
  kernel_size <- nn_util_pair(kernel_size)
  if (is.null(stride) || length(stride) == 0) {
    stride <- kernel_size
  } else {
    stride <- nn_util_pair(stride)
  }

  padding <- nn_util_pair(padding)

  output_size <- unpool_output_size(
    input, kernel_size, stride, padding,
    output_size
  )

  torch_max_unpool2d(input, indices, output_size)
}

#' Max_unpool3d
#'
#' Computes a partial inverse of `MaxPool3d`.
#'
#' @inheritParams nnf_max_unpool1d
#'
#' @export
nnf_max_unpool3d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL) {
  kernel_size <- nn_util_triple(kernel_size)
  padding <- nn_util_triple(padding)
  if (is.null(stride) || length(stride) == 0) {
    stride <- kernel_size
  } else {
    stride <- nn_util_triple(stride)
  }

  output_size <- unpool_output_size(
    input, kernel_size, stride, padding,
    output_size
  )

  torch_max_unpool3d(input, indices, output_size, stride, padding)
}

#' Fractional_max_pool2d
#'
#' Applies 2D fractional max pooling over an input signal composed of several input planes.
#'
#' Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham
#'
#' The max-pooling operation is applied in \eqn{kH * kW} regions by a stochastic
#' step size determined by the target output size.
#' The number of output features is equal to the number of input planes.
#'
#' @param input the input tensor
#' @param kernel_size the size of the window to take a max over. Can be a
#'   single number \eqn{k} (for a square kernel of \eqn{k * k}) or
#'   a tuple `(kH, kW)`
#' @param output_size the target output size of the image of the form \eqn{oH * oW}.
#'   Can be a tuple `(oH, oW)` or a single number \eqn{oH} for a square image \eqn{oH * oH}
#' @param output_ratio If one wants to have an output size as a ratio of the input size,
#'   this option can be given. This has to be a number or tuple in the range (0, 1)
#' @param return_indices if ``True``, will return the indices along with the outputs.
#' @param random_samples optional random samples.
#'
#' @export
nnf_fractional_max_pool2d <- function(input, kernel_size, output_size = NULL,
                                      output_ratio = NULL, return_indices = FALSE,
                                      random_samples = NULL) {
  if (is.null(output_size)) {
    output_ratio_ <- nn_util_pair(output_ratio)
    output_size <- list(
      as.integer(input$size(3) * output_ratio_[[1]]),
      as.integer(input$size(4) * output_ratio_[[2]])
    )
  }

  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(1), input$size(2), 2,
      dtype = input$dtype, device = input$device
    )
  }

  res <- torch_fractional_max_pool2d(
    self = input, kernel_size = kernel_size,
    output_size = output_size, random_samples = random_samples
  )

  if (return_indices) {
    res
  } else {
    res[[1]]
  }
}

#' Fractional_max_pool3d
#'
#' Applies 3D fractional max pooling over an input signal composed of several input planes.
#'
#' Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham
#'
#' The max-pooling operation is applied in \eqn{kT * kH * kW} regions by a stochastic
#' step size determined by the target output size.
#' The number of output features is equal to the number of input planes.
#'
#' @param input the input tensor
#' @param kernel_size the size of the window to take a max over. Can be a single number \eqn{k}
#'   (for a square kernel of \eqn{k * k * k}) or a tuple `(kT, kH, kW)`
#' @param output_size the target output size of the form \eqn{oT * oH * oW}.
#'   Can be a tuple `(oT, oH, oW)` or a single number \eqn{oH} for a cubic output
#'   \eqn{oH * oH * oH}
#' @param output_ratio If one wants to have an output size as a ratio of the
#'   input size, this option can be given. This has to be a number or tuple in the
#'   range (0, 1)
#' @param return_indices if ``True``, will return the indices along with the outputs.
#' @param random_samples undocumented argument.
#'
#' @export
nnf_fractional_max_pool3d <- function(input, kernel_size, output_size = NULL, output_ratio = NULL,
                                      return_indices = FALSE, random_samples = NULL) {
  if (is.null(output_size)) {
    if (is.null(output_ratio)) {
      value_error("output_ratio should not be NULL if output_size is NULL")
    }

    output_ratio_ <- nn_util_triple(output_ratio)
    output_size <- c(
      input$size(3) * output_ratio_[1],
      input$size(4) * output_ratio_[2],
      input$size(5) * output_ratio_[3]
    )
  }

  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(1), input$size(2), 3,
      dtype = input$dtype,
      device = input$device
    )
  }

  res <- torch_fractional_max_pool3d(
    self = input,
    kernel_size = kernel_size,
    output_size = output_size,
    random_samples = random_samples
  )


  if (return_indices) {
    res
  } else {
    res[[1]]
  }
}

#' Lp_pool1d
#'
#' Applies a 1D power-average pooling over an input signal composed of
#' several input planes. If the sum of all inputs to the power of `p` is
#' zero, the gradient is set to zero as well.
#'
#' @param input the input tensor
#' @param norm_type if inf than one gets max pooling if 0 you get sum pooling (
#'   proportional to the avg pooling)
#' @param kernel_size a single int, the size of the window
#' @param stride a single int, the stride of the window. Default value is kernel_size
#' @param ceil_mode when True, will use ceil instead of floor to compute the output shape
#'
#' @export
nnf_lp_pool1d <- function(input, norm_type, kernel_size, stride = NULL,
                          ceil_mode = FALSE) {
  if (!is.null(stride) || length(stride) == 0) {
    out <- nnf_avg_pool1d(input$pow(norm_type), kernel_size, stride, 0, ceil_mode)
  } else {
    out <- nnf_avg_pool1d(input$pow(norm_type), kernel_size,
      padding = 0,
      ceil_mode = ceil_mode
    )
  }

  (torch_sign(out) * nnf_relu(torch_abs(out)))$mul(kernel_size)$pow(1 / norm_type)
}

#' Lp_pool2d
#'
#' Applies a 2D power-average pooling over an input signal composed of
#' several input planes. If the sum of all inputs to the power of `p` is
#' zero, the gradient is set to zero as well.
#'
#' @inheritParams nnf_lp_pool1d
#'
#' @export
nnf_lp_pool2d <- function(input, norm_type, kernel_size, stride = NULL,
                          ceil_mode = FALSE) {
  k <- nn_util_pair(kernel_size)
  if (!length(stride) == 0) {
    out <- nnf_avg_pool2d(input$pow(norm_type), kernel_size, stride, 0, ceil_mode)
  } else {
    out <- nnf_avg_pool2d(input$pow(norm_type), kernel_size,
      padding = 0,
      ceil_mode = ceil_mode
    )
  }

  (torch_sign(out) * nnf_relu(torch_abs(out)))$mul(k[[1]] * k[[2]])$pow(1 / norm_type)
}
