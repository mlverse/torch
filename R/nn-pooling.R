#' @include nn.R
NULL

nn_max_pool_nd <- nn_module(
  "nn_max_pool_nd",
  initialize = function(kernel_size, stride = NULL, padding = 0, dilation = 1,
                        return_indices = FALSE, ceil_mode = FALSE) {
    self$kernel_size <- kernel_size

    if (is.null(stride)) {
      self$stride <- kernel_size
    } else {
      self$stride <- stride
    }

    self$padding <- padding
    self$dilation <- dilation
    self$return_indices <- return_indices
    self$ceil_mode <- ceil_mode
  }
)

#' MaxPool1D module
#'
#' Applies a 1D max pooling over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, L)}
#' and output \eqn{(N, C, L_{out})} can be precisely described as:
#'
#' \deqn{
#'   out(N_i, C_j, k) = \max_{m=0, \ldots, \mbox{kernel\_size} - 1}
#' input(N_i, C_j, stride \times k + m)
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on both sides
#' for `padding` number of points. `dilation` controls the spacing between the kernel points.
#' It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
#' has a nice visualization of what `dilation` does.
#'
#' @param kernel_size the size of the window to take a max over
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on both sides
#' @param dilation a parameter that controls the stride of elements in the window
#' @param return_indices if `TRUE`, will return the max indices along with the outputs.
#'    Useful for  `nn_max_unpool1d()` later.
#' @param ceil_mode when `TRUE`, will use `ceil` instead of `floor` to compute the output shape
#'
#' @section Shape:
#' - Input: \eqn{(N, C, L_{in})}
#' - Output: \eqn{(N, C, L_{out})}, where
#'
#' \deqn{
#'   L_{out} = \left\lfloor \frac{L_{in} + 2 \times \mbox{padding} - \mbox{dilation}
#'     \times (\mbox{kernel\_size} - 1) - 1}{\mbox{stride}} + 1\right\rfloor
#' }
#'
#' @examples
#' # pool of size=3, stride=2
#' m <- nn_max_pool1d(3, stride = 2)
#' input <- torch_randn(20, 16, 50)
#' output <- m(input)
#' @export
nn_max_pool1d <- nn_module(
  "nn_max_pool1d",
  inherit = nn_max_pool_nd,
  forward = function(input) {
    nnf_max_pool1d(
      input, self$kernel_size, self$stride,
      self$padding, self$dilation, self$ceil_mode,
      self$return_indices
    )
  }
)

#' MaxPool2D module
#'
#' Applies a 2D max pooling over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, H, W)},
#' output \eqn{(N, C, H_{out}, W_{out})} and `kernel_size` \eqn{(kH, kW)}
#' can be precisely described as:
#'
#' \deqn{
#' \begin{array}{ll}
#' out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
#' & \mbox{input}(N_i, C_j, \mbox{stride[0]} \times h + m,
#'                \mbox{stride[1]} \times w + n)
#' \end{array}
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on both sides
#' for `padding` number of points. `dilation` controls the spacing between the kernel points.
#' It is harder to describe, but this `link` has a nice visualization of what `dilation` does.
#'
#' The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:
#'
#' - a single `int` -- in which case the same value is used for the height and width dimension
#' - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
#'   and the second `int` for the width dimension
#'
#' @param kernel_size the size of the window to take a max over
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on both sides
#' @param dilation a parameter that controls the stride of elements in the window
#' @param return_indices if `TRUE`, will return the max indices along with the outputs.
#'   Useful for `nn_max_unpool2d()` later.
#' @param ceil_mode when `TRUE`, will use `ceil` instead of `floor` to compute the output shape
#'
#' @section Shape:
#' - Input: \eqn{(N, C, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, H_{out}, W_{out})}, where
#'
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in} + 2 * \mbox{padding[0]} - \mbox{dilation[0]}
#'     \times (\mbox{kernel\_size[0]} - 1) - 1}{\mbox{stride[0]}} + 1\right\rfloor
#' }
#'
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in} + 2 * \mbox{padding[1]} - \mbox{dilation[1]}
#'     \times (\mbox{kernel\_size[1]} - 1) - 1}{\mbox{stride[1]}} + 1\right\rfloor
#' }
#'
#' @examples
#' # pool of square window of size=3, stride=2
#' m <- nn_max_pool2d(3, stride = 2)
#' # pool of non-square window
#' m <- nn_max_pool2d(c(3, 2), stride = c(2, 1))
#' input <- torch_randn(20, 16, 50, 32)
#' output <- m(input)
#' @export
nn_max_pool2d <- nn_module(
  "nn_max_pool2d",
  inherit = nn_max_pool_nd,
  forward = function(input) {
    nnf_max_pool2d(
      input, self$kernel_size, self$stride,
      self$padding, self$dilation, self$ceil_mode,
      self$return_indices
    )
  }
)

#' Applies a 3D max pooling over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, D, H, W)},
#' output \eqn{(N, C, D_{out}, H_{out}, W_{out})} and `kernel_size` \eqn{(kD, kH, kW)}
#' can be precisely described as:
#'
#' \deqn{
#' \begin{array}{ll}
#' \mbox{out}(N_i, C_j, d, h, w) = & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
#'  & \mbox{input}(N_i, C_j, \mbox{stride[0]} \times d + k, \mbox{stride[1]} \times h + m, \mbox{stride[2]} \times w + n)
#' \end{array}
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on both sides
#' for `padding` number of points. `dilation` controls the spacing between the kernel points.
#' It is harder to describe, but this `link`_ has a nice visualization of what `dilation` does.
#' The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:
#'  - a single `int` -- in which case the same value is used for the depth, height and width dimension
#'  - a `tuple` of three ints -- in which case, the first `int` is used for the depth dimension,
#'    the second `int` for the height dimension and the third `int` for the width dimension
#'
#' @param kernel_size the size of the window to take a max over
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on all three sides
#' @param dilation a parameter that controls the stride of elements in the window
#' @param return_indices if `TRUE`, will return the max indices along with the outputs.
#'   Useful for `torch_nn.MaxUnpool3d` later
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#'
#' @section Shape:
#' - Input: \eqn{(N, C, D_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, D_{out}, H_{out}, W_{out})}, where
#' \deqn{
#'   D_{out} = \left\lfloor\frac{D_{in} + 2 \times \mbox{padding}[0] - \mbox{dilation}[0] \times
#'     (\mbox{kernel\_size}[0] - 1) - 1}{\mbox{stride}[0]} + 1\right\rfloor
#' }
#'
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in} + 2 \times \mbox{padding}[1] - \mbox{dilation}[1] \times
#'     (\mbox{kernel\_size}[1] - 1) - 1}{\mbox{stride}[1]} + 1\right\rfloor
#' }
#'
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in} + 2 \times \mbox{padding}[2] - \mbox{dilation}[2] \times
#'     (\mbox{kernel\_size}[2] - 1) - 1}{\mbox{stride}[2]} + 1\right\rfloor
#' }
#'
#' @examples
#' # pool of square window of size=3, stride=2
#' m <- nn_max_pool3d(3, stride = 2)
#' # pool of non-square window
#' m <- nn_max_pool3d(c(3, 2, 2), stride = c(2, 1, 2))
#' input <- torch_randn(20, 16, 50, 44, 31)
#' output <- m(input)
#' @export
nn_max_pool3d <- nn_module(
  "nn_max_pool3d",
  inherit = nn_max_pool_nd,
  forward = function(input) {
    nnf_max_pool3d(
      input, self$kernel_size, self$stride,
      self$padding, self$dilation, self$ceil_mode,
      self$return_indices
    )
  }
)

#' Computes a partial inverse of `MaxPool1d`.
#'
#' `MaxPool1d` is not fully invertible, since the non-maximal values are lost.
#' `MaxUnpool1d` takes in as input the output of `MaxPool1d`
#' including the indices of the maximal values and computes a partial inverse
#' in which all non-maximal values are set to zero.
#'
#' @note `MaxPool1d` can map several input sizes to the same output
#'   sizes. Hence, the inversion process can get ambiguous.
#'   To accommodate this, you can provide the needed output size
#'   as an additional argument `output_size` in the forward call.
#'   See the Inputs and Example below.
#'
#' @param kernel_size (int or tuple): Size of the max pooling window.
#' @param stride (int or tuple): Stride of the max pooling window.
#'   It is set to `kernel_size` by default.
#' @param padding (int or tuple): Padding that was added to the input
#'
#' @section Inputs:
#'
#' - `input`: the input Tensor to invert
#' - `indices`: the indices given out by [nn_max_pool1d()]
#' - `output_size` (optional): the targeted output size
#'
#' @section Shape:
#' - Input: \eqn{(N, C, H_{in})}
#' - Output: \eqn{(N, C, H_{out})}, where
#' \deqn{
#'   H_{out} = (H_{in} - 1) \times \mbox{stride}[0] - 2 \times \mbox{padding}[0] + \mbox{kernel\_size}[0]
#' }
#' or as given by `output_size` in the call operator
#'
#' @examples
#' pool <- nn_max_pool1d(2, stride = 2, return_indices = TRUE)
#' unpool <- nn_max_unpool1d(2, stride = 2)
#'
#' input <- torch_tensor(array(1:8 / 1, dim = c(1, 1, 8)))
#' out <- pool(input)
#' unpool(out[[1]], out[[2]])
#'
#' # Example showcasing the use of output_size
#' input <- torch_tensor(array(1:8 / 1, dim = c(1, 1, 8)))
#' out <- pool(input)
#' unpool(out[[1]], out[[2]], output_size = input$size())
#' unpool(out[[1]], out[[2]])
#' @export
nn_max_unpool1d <- nn_module(
  "nn_max_unpool1d",
  initialize = function(kernel_size, stride = NULL, padding = 0) {
    self$kernel_size <- nn_util_single(kernel_size)

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- nn_util_single(stride)
    self$padding <- nn_util_single(padding)
  },
  forward = function(input, indices, output_size = NULL) {
    nnf_max_unpool1d(
      input, indices, self$kernel_size, self$stride,
      self$padding, output_size
    )
  }
)


#' Computes a partial inverse of `MaxPool2d`.
#'
#' `MaxPool2d` is not fully invertible, since the non-maximal values are lost.
#' `MaxUnpool2d` takes in as input the output of `MaxPool2d`
#' including the indices of the maximal values and computes a partial inverse
#' in which all non-maximal values are set to zero.
#'
#' @note `MaxPool2d` can map several input sizes to the same output
#'   sizes. Hence, the inversion process can get ambiguous.
#'   To accommodate this, you can provide the needed output size
#'   as an additional argument `output_size` in the forward call.
#'   See the Inputs and Example below.
#'
#' @param kernel_size (int or tuple): Size of the max pooling window.
#' @param stride (int or tuple): Stride of the max pooling window.
#'   It is set to `kernel_size` by default.
#' @param padding (int or tuple): Padding that was added to the input
#'
#' @section Inputs:
#' - `input`: the input Tensor to invert
#' - `indices`: the indices given out by [nn_max_pool2d()]
#' - `output_size` (optional): the targeted output size
#'
#' @section Shape:
#' - Input: \eqn{(N, C, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, H_{out}, W_{out})}, where
#' \deqn{
#'   H_{out} = (H_{in} - 1) \times \mbox{stride[0]} - 2 \times \mbox{padding[0]} + \mbox{kernel\_size[0]}
#' }
#' \deqn{
#'   W_{out} = (W_{in} - 1) \times \mbox{stride[1]} - 2 \times \mbox{padding[1]} + \mbox{kernel\_size[1]}
#' }
#' or as given by `output_size` in the call operator
#'
#' @examples
#'
#' pool <- nn_max_pool2d(2, stride = 2, return_indices = TRUE)
#' unpool <- nn_max_unpool2d(2, stride = 2)
#' input <- torch_randn(1, 1, 4, 4)
#' out <- pool(input)
#' unpool(out[[1]], out[[2]])
#'
#' # specify a different output size than input size
#' unpool(out[[1]], out[[2]], output_size = c(1, 1, 5, 5))
#' @export
nn_max_unpool2d <- nn_module(
  "nn_max_unpool2d",
  initialize = function(kernel_size, stride = NULL, padding = 0) {
    self$kernel_size <- nn_util_pair(kernel_size)

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- nn_util_pair(stride)
    self$padding <- nn_util_pair(padding)
  },
  forward = function(input, indices, output_size = NULL) {
    nnf_max_unpool2d(
      input, indices, self$kernel_size, self$stride,
      self$padding, output_size
    )
  }
)


#' Computes a partial inverse of `MaxPool3d`.
#'
#' `MaxPool3d` is not fully invertible, since the non-maximal values are lost.
#' `MaxUnpool3d` takes in as input the output of `MaxPool3d`
#' including the indices of the maximal values and computes a partial inverse
#' in which all non-maximal values are set to zero.
#'
#' @note `MaxPool3d` can map several input sizes to the same output
#' sizes. Hence, the inversion process can get ambiguous.
#' To accommodate this, you can provide the needed output size
#' as an additional argument `output_size` in the forward call.
#' See the Inputs section below.
#'
#' @param kernel_size (int or tuple): Size of the max pooling window.
#' @param stride (int or tuple): Stride of the max pooling window.
#'   It is set to `kernel_size` by default.
#' @param padding (int or tuple): Padding that was added to the input
#'
#' @section Inputs:
#' - `input`: the input Tensor to invert
#' - `indices`: the indices given out by [nn_max_pool3d()]
#' - `output_size` (optional): the targeted output size
#'
#' @section Shape:
#' - Input: \eqn{(N, C, D_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, D_{out}, H_{out}, W_{out})}, where
#'
#' \deqn{
#'   D_{out} = (D_{in} - 1) \times \mbox{stride[0]} - 2 \times \mbox{padding[0]} + \mbox{kernel\_size[0]}
#' }
#' \deqn{
#'   H_{out} = (H_{in} - 1) \times \mbox{stride[1]} - 2 \times \mbox{padding[1]} + \mbox{kernel\_size[1]}
#' }
#' \deqn{
#'   W_{out} = (W_{in} - 1) \times \mbox{stride[2]} - 2 \times \mbox{padding[2]} + \mbox{kernel\_size[2]}
#' }
#'
#' or as given by `output_size` in the call operator
#'
#' @examples
#'
#' # pool of square window of size=3, stride=2
#' pool <- nn_max_pool3d(3, stride = 2, return_indices = TRUE)
#' unpool <- nn_max_unpool3d(3, stride = 2)
#' out <- pool(torch_randn(20, 16, 51, 33, 15))
#' unpooled_output <- unpool(out[[1]], out[[2]])
#' unpooled_output$size()
#' @export
nn_max_unpool3d <- nn_module(
  "nn_max_unpool3d",
  initialize = function(kernel_size, stride = NULL, padding = 0) {
    self$kernel_size <- nn_util_triple(kernel_size)

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- nn_util_triple(stride)
    self$padding <- nn_util_triple(padding)
  },
  forward = function(input, indices, output_size = NULL) {
    nnf_max_unpool3d(
      input, indices, self$kernel_size, self$stride,
      self$padding, output_size
    )
  }
)

#' Applies a 1D average pooling over an input signal composed of several
#' input planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, L)},
#' output \eqn{(N, C, L_{out})} and `kernel_size` \eqn{k}
#' can be precisely described as:
#'
#' \deqn{
#'   \mbox{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
#' \mbox{input}(N_i, C_j, \mbox{stride} \times l + m)
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on both sides
#' for `padding` number of points.
#'
#' The parameters `kernel_size`, `stride`, `padding` can each be
#' an `int` or a one-element tuple.
#'
#' @param kernel_size the size of the window
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on both sides
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#' @param count_include_pad when TRUE, will include the zero-padding in the averaging calculation
#'
#' @section Shape:
#' - Input: \eqn{(N, C, L_{in})}
#' - Output: \eqn{(N, C, L_{out})}, where
#'
#' \deqn{
#'   L_{out} = \left\lfloor \frac{L_{in} +
#'       2 \times \mbox{padding} - \mbox{kernel\_size}}{\mbox{stride}} + 1\right\rfloor
#' }
#'
#' @examples
#'
#' # pool with window of size=3, stride=2
#' m <- nn_avg_pool1d(3, stride = 2)
#' m(torch_randn(1, 1, 8))
#' @export
nn_avg_pool1d <- nn_module(
  "nn_avg_pool1d",
  initialize = function(kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                        count_include_pad = TRUE) {
    self$kernel_size <- nn_util_single(kernel_size)

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- nn_util_single(stride)

    self$padding <- nn_util_single(padding)
    self$ceil_mode <- ceil_mode
    self$count_include_pad <- count_include_pad
  },
  forward = function(input) {
    nnf_avg_pool1d(
      input, self$kernel_size, self$stride, self$padding, self$ceil_mode,
      self$count_include_pad
    )
  }
)

#' Applies a 2D average pooling over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, H, W)},
#' output \eqn{(N, C, H_{out}, W_{out})} and `kernel_size` \eqn{(kH, kW)}
#' can be precisely described as:
#'
#' \deqn{
#'   out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
#' input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on both sides
#' for `padding` number of points.
#'
#' The parameters `kernel_size`, `stride`, `padding` can either be:
#'
#' - a single `int` -- in which case the same value is used for the height and width dimension
#' - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
#' and the second `int` for the width dimension
#'
#'
#' @param kernel_size the size of the window
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on both sides
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#' @param count_include_pad when TRUE, will include the zero-padding in the averaging calculation
#' @param divisor_override if specified, it will be used as divisor, otherwise `kernel_size` will be used
#'
#' @section Shape:
#' - Input: \eqn{(N, C, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, H_{out}, W_{out})}, where
#'
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \mbox{padding}[0] -
#'       \mbox{kernel\_size}[0]}{\mbox{stride}[0]} + 1\right\rfloor
#' }
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \mbox{padding}[1] -
#'       \mbox{kernel\_size}[1]}{\mbox{stride}[1]} + 1\right\rfloor
#' }
#'
#' @examples
#'
#' # pool of square window of size=3, stride=2
#' m <- nn_avg_pool2d(3, stride = 2)
#' # pool of non-square window
#' m <- nn_avg_pool2d(c(3, 2), stride = c(2, 1))
#' input <- torch_randn(20, 16, 50, 32)
#' output <- m(input)
#' @export
nn_avg_pool2d <- nn_module(
  "nn_avg_pool2d",
  initialize = function(kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                        count_include_pad = TRUE, divisor_override = NULL) {
    self$kernel_size <- kernel_size

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- stride

    self$padding <- padding
    self$ceil_mode <- ceil_mode
    self$count_include_pad <- count_include_pad
    self$divisor_override <- divisor_override
  },
  forward = function(input) {
    nnf_avg_pool2d(
      input, self$kernel_size, self$stride, self$padding, self$ceil_mode,
      self$count_include_pad, self$divisor_override
    )
  }
)

#' Applies a 3D average pooling over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size \eqn{(N, C, D, H, W)},
#' output \eqn{(N, C, D_{out}, H_{out}, W_{out})} and `kernel_size` \eqn{(kD, kH, kW)}
#' can be precisely described as:
#'
#' \deqn{
#' \begin{array}{ll}
#' \mbox{out}(N_i, C_j, d, h, w) = & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
#' & \frac{\mbox{input}(N_i, C_j, \mbox{stride}[0] \times d + k, \mbox{stride}[1] \times h + m, \mbox{stride}[2] \times w + n)}{kD \times kH \times kW}
#' \end{array}
#' }
#'
#' If `padding` is non-zero, then the input is implicitly zero-padded on all three sides
#' for `padding` number of points.
#'
#' The parameters `kernel_size`, `stride` can either be:
#'
#' - a single `int` -- in which case the same value is used for the depth, height and width dimension
#' - a `tuple` of three ints -- in which case, the first `int` is used for the depth dimension,
#' the second `int` for the height dimension and the third `int` for the width dimension
#'
#' @param kernel_size the size of the window
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param padding implicit zero padding to be added on all three sides
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#' @param count_include_pad when TRUE, will include the zero-padding in the averaging calculation
#' @param divisor_override if specified, it will be used as divisor, otherwise `kernel_size` will be used
#'
#' @section Shape:
#' - Input: \eqn{(N, C, D_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, D_{out}, H_{out}, W_{out})}, where
#'
#' \deqn{
#'   D_{out} = \left\lfloor\frac{D_{in} + 2 \times \mbox{padding}[0] -
#'       \mbox{kernel\_size}[0]}{\mbox{stride}[0]} + 1\right\rfloor
#' }
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in} + 2 \times \mbox{padding}[1] -
#'       \mbox{kernel\_size}[1]}{\mbox{stride}[1]} + 1\right\rfloor
#' }
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in} + 2 \times \mbox{padding}[2] -
#'       \mbox{kernel\_size}[2]}{\mbox{stride}[2]} + 1\right\rfloor
#' }
#'
#' @examples
#'
#' # pool of square window of size=3, stride=2
#' m <- nn_avg_pool3d(3, stride = 2)
#' # pool of non-square window
#' m <- nn_avg_pool3d(c(3, 2, 2), stride = c(2, 1, 2))
#' input <- torch_randn(20, 16, 50, 44, 31)
#' output <- m(input)
#' @export
nn_avg_pool3d <- nn_module(
  "nn_avg_pool3d",
  initialize = function(kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE,
                        count_include_pad = TRUE, divisor_override = NULL) {
    self$kernel_size <- kernel_size

    if (is.null(stride)) {
      stride <- kernel_size
    }

    self$stride <- stride

    self$padding <- padding
    self$ceil_mode <- ceil_mode
    self$count_include_pad <- count_include_pad
    self$divisor_override <- divisor_override
  },
  forward = function(input) {
    nnf_avg_pool3d(
      input, self$kernel_size, self$stride, self$padding, self$ceil_mode,
      self$count_include_pad, self$divisor_override
    )
  }
)

#' Applies a 2D fractional max pooling over an input signal composed of several input planes.
#'
#' Fractional MaxPooling is described in detail in the paper
#' [Fractional MaxPooling](https://arxiv.org/abs/1412.6071) by Ben Graham
#'
#' The max-pooling operation is applied in \eqn{kH \times kW} regions by a stochastic
#' step size determined by the target output size.
#' The number of output features is equal to the number of input planes.
#'
#' @param kernel_size the size of the window to take a max over.
#'   Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
#' @param output_size the target output size of the image of the form `oH x oW`.
#'   Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`
#' @param output_ratio If one wants to have an output size as a ratio of the input size, this option can be given.
#'   This has to be a number or tuple in the range (0, 1)
#' @param return_indices if `TRUE`, will return the indices along with the outputs.
#'   Useful to pass to [nn_max_unpool2d()]. Default: `FALSE`
#'
#' @examples
#' # pool of square window of size=3, and target output size 13x12
#' m <- nn_fractional_max_pool2d(3, output_size = c(13, 12))
#' # pool of square window and target output size being half of input image size
#' m <- nn_fractional_max_pool2d(3, output_ratio = c(0.5, 0.5))
#' input <- torch_randn(20, 16, 50, 32)
#' output <- m(input)
#' @export
nn_fractional_max_pool2d <- nn_module(
  "nn_fractional_max_pool2d",
  initialize = function(kernel_size, output_size = NULL,
                        output_ratio = NULL,
                        return_indices = FALSE) {
    random_samples <- NULL

    self$kernel_size <- nn_util_pair(kernel_size)
    self$return_indices <- return_indices
    self$register_buffer("random_samples", random_samples)

    if (!is.null(output_size)) {
      output_size <- nn_util_pair(output_size)
    }

    self$output_size <- output_size

    if (!is.null(output_ratio)) {
      output_ratio <- nn_util_pair(output_ratio)
    }

    self$output_ratio <- output_ratio


    if (is.null(output_ratio) && is.null(output_size)) {
      value_error("both output_size and output_ratio are NULL")
    }

    if (!is.null(output_ratio) && !is.null(output_size)) {
      value_error("both output_size and oytput_ratio are not NULL")
    }

    if (!is.null(output_ratio)) {
      if (any(output_ratio > 1 | output_ratio < 0)) {
        value_error("output_ratio must be between 0 and 1.")
      }
    }
  },
  forward = function(input) {
    nnf_fractional_max_pool2d(
      input, self$kernel_size, self$output_size, self$output_ratio,
      self$return_indices,
      random_samples = self$random_samples
    )
  }
)

#' Applies a 3D fractional max pooling over an input signal composed of several input planes.
#'
#' Fractional MaxPooling is described in detail in the paper
#' [Fractional MaxPooling](https://arxiv.org/abs/1412.6071) by Ben Graham
#'
#' The max-pooling operation is applied in \eqn{kTxkHxkW} regions by a stochastic
#' step size determined by the target output size.
#' The number of output features is equal to the number of input planes.
#'
#' @param kernel_size the size of the window to take a max over.
#'   Can be a single number k (for a square kernel of k x k x k) or a tuple `(kt x kh x kw)`
#' @param output_size the target output size of the image of the form `oT x oH x oW`.
#'   Can be a tuple `(oT, oH, oW)` or a single number oH for a square image `oH x oH x oH`
#' @param output_ratio If one wants to have an output size as a ratio of the input size, this option can be given.
#'   This has to be a number or tuple in the range (0, 1)
#' @param return_indices if `TRUE`, will return the indices along with the outputs.
#'   Useful to pass to [nn_max_unpool3d()]. Default: `FALSE`
#'
#' @examples
#' # pool of cubic window of size=3, and target output size 13x12x11
#' m <- nn_fractional_max_pool3d(3, output_size = c(13, 12, 11))
#' # pool of cubic window and target output size being half of input size
#' m <- nn_fractional_max_pool3d(3, output_ratio = c(0.5, 0.5, 0.5))
#' input <- torch_randn(20, 16, 50, 32, 16)
#' output <- m(input)
#' @export
nn_fractional_max_pool3d <- nn_module(
  "nn_fractional_max_pool3d",
  initialize = function(kernel_size, output_size = NULL,
                        output_ratio = NULL,
                        return_indices = FALSE) {
    random_samples <- NULL

    self$kernel_size <- nn_util_triple(kernel_size)
    self$return_indices <- return_indices
    self$register_buffer("random_samples", random_samples)

    if (!is.null(output_size)) {
      output_size <- nn_util_triple(output_size)
    }

    self$output_size <- output_size

    if (!is.null(output_ratio)) {
      output_ratio <- nn_util_triple(output_ratio)
    }

    self$output_ratio <- output_ratio


    if (is.null(output_ratio) && is.null(output_size)) {
      value_error("both output_size and output_ratio are NULL")
    }

    if (!is.null(output_ratio) && !is.null(output_size)) {
      value_error("both output_size and oytput_ratio are not NULL")
    }

    if (!is.null(output_ratio)) {
      if (any(output_ratio > 1 | output_ratio < 0)) {
        value_error("output_ratio must be between 0 and 1.")
      }
    }
  },
  forward = function(input) {
    nnf_fractional_max_pool3d(
      input, self$kernel_size, self$output_size, self$output_ratio,
      self$return_indices,
      random_samples = self$random_samples
    )
  }
)

lp_pool_nd <- nn_module(
  "lp_pool_nd",
  initialize = function(norm_type, kernel_size, stride = NULL,
                        ceil_mode = FALSE) {
    self$norm_type <- norm_type
    self$kernel_size <- kernel_size
    self$stride <- stride
    self$ceil_mode <- ceil_mode
  }
)

#' Applies a 1D power-average pooling over an input signal composed of several input
#' planes.
#'
#' On each window, the function computed is:
#'
#' \deqn{
#'   f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}
#' }
#'
#' - At p = \eqn{\infty}, one gets Max Pooling
#' - At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)
#'
#' @note If the sum to the power of `p` is zero, the gradient of this function is
#' not defined. This implementation will set the gradient to zero in this case.
#'
#' @param norm_type if inf than one gets max pooling if 0 you get sum pooling (
#'   proportional to the avg pooling)
#' @param kernel_size a single int, the size of the window
#' @param stride a single int, the stride of the window. Default value is `kernel_size`
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#'
#' @section Shape:
#' - Input: \eqn{(N, C, L_{in})}
#' - Output: \eqn{(N, C, L_{out})}, where
#'
#' \deqn{
#'   L_{out} = \left\lfloor\frac{L_{in} - \mbox{kernel\_size}}{\mbox{stride}} + 1\right\rfloor
#' }
#'
#' @examples
#' # power-2 pool of window of length 3, with stride 2.
#' m <- nn_lp_pool1d(2, 3, stride = 2)
#' input <- torch_randn(20, 16, 50)
#' output <- m(input)
#' @export
nn_lp_pool1d <- nn_module(
  "nn_lp_pool1d",
  inherit = lp_pool_nd,
  forward = function(input) {
    nnf_lp_pool1d(
      input, self$norm_type, self$kernel_size,
      self$stride, self$ceil_mode
    )
  }
)

#' Applies a 2D power-average pooling over an input signal composed of several input
#' planes.
#'
#' On each window, the function computed is:
#'
#' \deqn{
#'   f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}
#' }
#'
#' - At p = \eqn{\infty}, one gets Max Pooling
#' - At p = 1, one gets Sum Pooling (which is proportional to average pooling)
#'
#' The parameters `kernel_size`, `stride` can either be:
#'
#' - a single `int` -- in which case the same value is used for the height and width dimension
#' - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
#' and the second `int` for the width dimension
#'
#' @note If the sum to the power of `p` is zero, the gradient of this function is
#' not defined. This implementation will set the gradient to zero in this case.
#'
#' @param norm_type if inf than one gets max pooling if 0 you get sum pooling (
#'   proportional to the avg pooling)
#' @param kernel_size the size of the window
#' @param stride the stride of the window. Default value is `kernel_size`
#' @param ceil_mode when TRUE, will use `ceil` instead of `floor` to compute the output shape
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, H_{in}, W_{in})}
#' - Output: \eqn{(N, C, H_{out}, W_{out})}, where
#'
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in} - \mbox{kernel\_size}[0]}{\mbox{stride}[0]} + 1\right\rfloor
#' }
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in} - \mbox{kernel\_size}[1]}{\mbox{stride}[1]} + 1\right\rfloor
#' }
#'
#' @examples
#'
#' # power-2 pool of square window of size=3, stride=2
#' m <- nn_lp_pool2d(2, 3, stride = 2)
#' # pool of non-square window of power 1.2
#' m <- nn_lp_pool2d(1.2, c(3, 2), stride = c(2, 1))
#' input <- torch_randn(20, 16, 50, 32)
#' output <- m(input)
#' @export
nn_lp_pool2d <- nn_module(
  "nn_lp_pool2d",
  inherit = lp_pool_nd,
  forward = function(input) {
    nnf_lp_pool2d(
      input, self$norm_type, self$kernel_size,
      self$stride, self$ceil_mode
    )
  }
)

#' Applies a 1D adaptive max pooling over an input signal composed of several input planes.
#'
#' The output size is H, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#' @param output_size the target output size H
#' @param return_indices if `TRUE`, will return the indices along with the outputs.
#'   Useful to pass to [nn_max_unpool1d()]. Default: `FALSE`
#'
#' @examples
#' # target output size of 5
#' m <- nn_adaptive_max_pool1d(5)
#' input <- torch_randn(1, 64, 8)
#' output <- m(input)
#' @export
nn_adaptive_max_pool1d <- nn_module(
  "nn_adaptive_max_pool1d",
  initialize = function(output_size, return_indices = FALSE) {
    self$output_size <- output_size
    self$return_indices <- return_indices
  },
  forward = function(input) {
    nnf_adaptive_max_pool1d(input, self$output_size, self$return_indices)
  }
)

#' Applies a 2D adaptive max pooling over an input signal composed of several input planes.
#'
#' The output is of size H x W, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#' @param output_size the target output size of the image of the form H x W.
#'   Can be a tuple `(H, W)` or a single H for a square image H x H.
#'   H and W can be either a `int`, or `None` which means the size will
#'   be the same as that of the input.
#' @param return_indices if `TRUE`, will return the indices along with the outputs.
#'   Useful to pass to [nn_max_unpool2d()]. Default: `FALSE`
#'
#' @examples
#' # target output size of 5x7
#' m <- nn_adaptive_max_pool2d(c(5, 7))
#' input <- torch_randn(1, 64, 8, 9)
#' output <- m(input)
#' # target output size of 7x7 (square)
#' m <- nn_adaptive_max_pool2d(7)
#' input <- torch_randn(1, 64, 10, 9)
#' output <- m(input)
#' @export
nn_adaptive_max_pool2d <- nn_module(
  "nn_adaptive_max_pool2d",
  initialize = function(output_size, return_indices = FALSE) {
    self$output_size <- nn_util_pair(output_size)
    self$return_indices <- return_indices
  },
  forward = function(input) {
    nnf_adaptive_max_pool2d(input, self$output_size, self$return_indices)
  }
)

#' Applies a 3D adaptive max pooling over an input signal composed of several input planes.
#'
#' The output is of size D x H x W, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#'
#' @param output_size the target output size of the image of the form D x H x W.
#'   Can be a tuple (D, H, W) or a single D for a cube D x D x D.
#'   D, H and W can be either a `int`, or `None` which means the size will
#'   be the same as that of the input.
#' @param return_indices if `TRUE`, will return the indices along with the outputs.
#'   Useful to pass to [nn_max_unpool3d()]. Default: `FALSE`
#'
#' @examples
#' # target output size of 5x7x9
#' m <- nn_adaptive_max_pool3d(c(5, 7, 9))
#' input <- torch_randn(1, 64, 8, 9, 10)
#' output <- m(input)
#' # target output size of 7x7x7 (cube)
#' m <- nn_adaptive_max_pool3d(7)
#' input <- torch_randn(1, 64, 10, 9, 8)
#' output <- m(input)
#' @export
nn_adaptive_max_pool3d <- nn_module(
  "nn_adaptive_max_pool3d",
  initialize = function(output_size, return_indices = FALSE) {
    self$output_size <- nn_util_triple(output_size)
    self$return_indices <- return_indices
  },
  forward = function(input) {
    nnf_adaptive_max_pool3d(input, self$output_size, self$return_indices)
  }
)

#' Applies a 1D adaptive average pooling over an input signal composed of several input planes.
#'
#' The output size is H, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#' @param output_size the target output size H
#'
#' @examples
#' # target output size of 5
#' m <- nn_adaptive_avg_pool1d(5)
#' input <- torch_randn(1, 64, 8)
#' output <- m(input)
#' @export
nn_adaptive_avg_pool1d <- nn_module(
  "nn_adaptive_avg_pool1d",
  initialize = function(output_size) {
    self$output_size <- output_size
  },
  forward = function(input) {
    nnf_adaptive_avg_pool1d(input, self$output_size)
  }
)

#' Applies a 2D adaptive average pooling over an input signal composed of several input planes.
#'
#' The output is of size H x W, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#' @param output_size the target output size of the image of the form H x W.
#'   Can be a tuple (H, W) or a single H for a square image H x H.
#'   H and W can be either a `int`, or `NULL` which means the size will
#'   be the same as that of the input.
#'
#' @examples
#' # target output size of 5x7
#' m <- nn_adaptive_avg_pool2d(c(5, 7))
#' input <- torch_randn(1, 64, 8, 9)
#' output <- m(input)
#' # target output size of 7x7 (square)
#' m <- nn_adaptive_avg_pool2d(7)
#' input <- torch_randn(1, 64, 10, 9)
#' output <- m(input)
#' @export
nn_adaptive_avg_pool2d <- nn_module(
  "nn_adaptive_avg_pool2d",
  initialize = function(output_size) {
    self$output_size <- nn_util_pair(output_size)
  },
  forward = function(input) {
    nnf_adaptive_avg_pool2d(input, self$output_size)
  }
)

#' Applies a 3D adaptive average pooling over an input signal composed of several input planes.
#'
#' The output is of size D x H x W, for any input size.
#' The number of output features is equal to the number of input planes.
#'
#' @param output_size the target output size of the form D x H x W.
#'   Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
#'   D, H and W can be either a `int`, or `None` which means the size will
#'   be the same as that of the input.
#'
#' @examples
#' # target output size of 5x7x9
#' m <- nn_adaptive_avg_pool3d(c(5, 7, 9))
#' input <- torch_randn(1, 64, 8, 9, 10)
#' output <- m(input)
#' # target output size of 7x7x7 (cube)
#' m <- nn_adaptive_avg_pool3d(7)
#' input <- torch_randn(1, 64, 10, 9, 8)
#' output <- m(input)
#' @export
nn_adaptive_avg_pool3d <- nn_module(
  "nn_adaptive_avg_pool3d",
  initialize = function(output_size) {
    self$output_size <- nn_util_triple(output_size)
  },
  forward = function(input) {
    nnf_adaptive_avg_pool3d(input, self$output_size)
  }
)
