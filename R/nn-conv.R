#' @include nn.R
NULL

nn_conv_nd <- nn_module(
  "nn_conv_nd",
  initialize = function(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, transposed, output_padding,
                        groups, bias, padding_mode) {
    if (in_channels %% groups != 0) {
      value_error("in_channels must be divisable by groups")
    }

    if (out_channels %% groups != 0) {
      value_error("out_channels must be divisable by groups")
    }

    valid_padding_modes <- c("zeros", "reflect", "replicate", "circular")

    if (!padding_mode %in% valid_padding_modes) {
      value_error(
        "padding_mode must be one of [{paste(valid_padding_modes, collapse = ', ')}],",
        "but got padding_mode='{padding_mode}'."
      )
    }

    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$kernel_size <- kernel_size
    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation
    self$transposed <- transposed
    self$output_padding <- output_padding
    self$groups <- groups
    self$padding_mode <- padding_mode

    # `_reversed_padding_repeated_twice` is the padding to be passed to
    # `nnf_pad` if needed (e.g., for non-zero padding types that are
    # implemented as two ops: padding + conv). `nnf_pad` accepts paddings in
    # reverse order than the dimension.
    self$reversed_padding_repeated_twice_ <- nn_util_reverse_repeat_tuple(self$padding, 2)

    if (transposed) {
      self$weight <- nn_parameter(torch_empty(
        in_channels, out_channels %/% groups, !!!kernel_size
      ))
    } else {
      self$weight <- nn_parameter(torch_empty(
        out_channels, in_channels %/% groups, !!!kernel_size
      ))
    }

    if (bias) {
      self$bias <- nn_parameter(torch_empty(out_channels))
    } else {
      self$register_parameter("bias", NULL)
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    if (!is.null(self$bias)) {
      fans <- nn_init_calculate_fan_in_and_fan_out(self$weight)
      bound <- 1 / sqrt(fans[[1]])
      nn_init_uniform_(self$bias, -bound, bound)
    }
  }
)

#' Conv1D module
#'
#' Applies a 1D convolution over an input signal composed of several input
#' planes.
#' In the simplest case, the output value of the layer with input size
#' \eqn{(N, C_{\mbox{in}}, L)} and output \eqn{(N, C_{\mbox{out}}, L_{\mbox{out}})} can be
#' precisely described as:
#'
#' \deqn{
#' \mbox{out}(N_i, C_{\mbox{out}_j}) = \mbox{bias}(C_{\mbox{out}_j}) +
#'   \sum_{k = 0}^{C_{in} - 1} \mbox{weight}(C_{\mbox{out}_j}, k)
#' \star \mbox{input}(N_i, k)
#' }
#'
#' where \eqn{\star} is the valid
#' [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) operator,
#' \eqn{N} is a batch size, \eqn{C} denotes a number of channels,
#' \eqn{L} is a length of signal sequence.
#'
#' * `stride` controls the stride for the cross-correlation, a single
#'   number or a one-element tuple.
#' * `padding` controls the amount of implicit zero-paddings on both sides
#'   for `padding` number of points.
#' * `dilation` controls the spacing between the kernel points; also
#'   known as the à trous algorithm. It is harder to describe, but this
#'   [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
#'   has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#'   `in_channels` and `out_channels` must both be divisible by
#'   `groups`. For example,
#'    * At groups=1, all inputs are convolved to all outputs.
#'    * At groups=2, the operation becomes equivalent to having two conv
#'    layers side by side, each seeing half the input channels,
#'    and producing half the output channels, and both subsequently
#'    concatenated.
#'    * At groups= `in_channels`, each input channel is convolved with
#'    its own set of filters,
#'    of size \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}.
#'
#' @section Note:
#'
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid
#' `cross-correlation`_, and not a full `cross-correlation`_.
#' It is up to the user to add proper padding.
#'
#' When `groups == in_channels` and `out_channels == K * in_channels`,
#' where `K` is a positive integer, this operation is also termed in
#' literature as depthwise convolution.
#' In other words, for an input of size \eqn{(N, C_{in}, L_{in})},
#' a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
#' \eqn{(C_{\mbox{in}}=C_{in}, C_{\mbox{out}}=C_{in} \times K, ..., \mbox{groups}=C_{in})}.
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int, tuple or str, optional) – Padding added to both sides of
#'   the input. Default: 0
#' @param padding_mode (string, optional): `'zeros'`, `'reflect'`,
#'   `'replicate'` or `'circular'`. Default: `'zeros'`
#' @param dilation (int or tuple, optional): Spacing between kernel
#'   elements. Default: 1
#' @param groups (int, optional): Number of blocked connections from input
#'   channels to output channels. Default: 1
#' @param bias (bool, optional): If `TRUE`, adds a learnable bias to the
#'   output. Default: `TRUE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C_{in}, L_{in})}
#' - Output: \eqn{(N, C_{out}, L_{out})} where
#'
#' \deqn{
#'   L_{out} = \left\lfloor\frac{L_{in} + 2 \times \mbox{padding} - \mbox{dilation}
#'     \times (\mbox{kernel\_size} - 1) - 1}{\mbox{stride}} + 1\right\rfloor
#' }
#'
#' @section Attributes:
#'
#' - weight (Tensor): the learnable weights of the module of shape
#' \eqn{(\mbox{out\_channels}, \frac{\mbox{in\_channels}}{\mbox{groups}}, \mbox{kernel\_size})}.
#' The values of these weights are sampled from
#' \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{in}} * \mbox{kernel\_size}}}
#' - bias (Tensor): the learnable bias of the module of shape
#' (out_channels). If `bias` is `TRUE`, then the values of these weights are
#' sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{in}} * \mbox{kernel\_size}}}
#'
#' @examples
#' m <- nn_conv1d(16, 33, 3, stride = 2)
#' input <- torch_randn(20, 16, 50)
#' output <- m(input)
#' @export
nn_conv1d <- nn_module(
  "nn_conv1d",
  inherit = nn_conv_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride = 1,
                        padding = 0, dilation = 1, groups = 1,
                        bias = TRUE, padding_mode = "zeros") {
    kernel_size <- nn_util_single(kernel_size)
    stride <- nn_util_single(stride)

    if (is.character(padding)) {
      padding <- padding
    } else {
      padding <- nn_util_single(padding)
    }

    dilation <- nn_util_single(dilation)
    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      FALSE, nn_util_single(0), groups, bias, padding_mode
    )
  },
  forward = function(input) {
    if (self$padding_mode != "zeros") {
      nnf_conv1d(
        nnf_pad(input, self$reversed_padding_repeated_twice_, mode = self$padding_mode),
        self$weight, self$bias, self$stride,
        nn_util_single(0), self$dilation, self$groups
      )
    } else {
      nnf_conv1d(
        input, self$weight, self$bias, self$stride,
        self$padding, self$dilation, self$groups
      )
    }
  }
)

#' Conv2D module
#'
#' Applies a 2D convolution over an input signal composed of several input
#' planes.
#'
#' In the simplest case, the output value of the layer with input size
#' \eqn{(N, C_{\mbox{in}}, H, W)} and output \eqn{(N, C_{\mbox{out}}, H_{\mbox{out}}, W_{\mbox{out}})}
#' can be precisely described as:
#'
#' \deqn{
#' \mbox{out}(N_i, C_{\mbox{out}_j}) = \mbox{bias}(C_{\mbox{out}_j}) +
#'   \sum_{k = 0}^{C_{\mbox{in}} - 1} \mbox{weight}(C_{\mbox{out}_j}, k) \star \mbox{input}(N_i, k)
#' }
#'
#' where \eqn{\star} is the valid 2D cross-correlation operator,
#' \eqn{N} is a batch size, \eqn{C} denotes a number of channels,
#' \eqn{H} is a height of input planes in pixels, and \eqn{W} is
#' width in pixels.
#'
#' * `stride` controls the stride for the cross-correlation, a single
#'   number or a tuple.
#' * `padding` controls the amount of implicit zero-paddings on both
#'   sides for `padding` number of points for each dimension.
#' * `dilation` controls the spacing between the kernel points; also
#'   known as the à trous algorithm. It is harder to describe, but this `link`_
#'   has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#'   `in_channels` and `out_channels` must both be divisible by
#'   `groups`. For example,
#'    * At groups=1, all inputs are convolved to all outputs.
#'    * At groups=2, the operation becomes equivalent to having two conv
#'      layers side by side, each seeing half the input channels,
#'      and producing half the output channels, and both subsequently
#'      concatenated.
#'    * At groups= `in_channels`, each input channel is convolved with
#'      its own set of filters, of size:
#'      \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}.
#'
#' The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:
#'
#' - a single `int` -- in which case the same value is used for the height and
#'   width dimension
#' - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
#'   and the second `int` for the width dimension
#'
#' @section Note:
#'
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid cross-correlation,
#' and not a full cross-correlation.
#' It is up to the user to add proper padding.
#'
#' When `groups == in_channels` and `out_channels == K * in_channels`,
#' where `K` is a positive integer, this operation is also termed in
#' literature as depthwise convolution.
#' In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
#' a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
#' \eqn{(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})}.
#'
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting `backends_cudnn_deterministic = TRUE`.
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int or tuple or string, optional): Zero-padding added to both sides of
#'   the input. controls the amount of padding applied to the input. It
#'   can be either a string `'valid'`, `'same'` or a tuple of ints giving the
#'   amount of implicit padding applied on both sides. Default: 0
#' @param padding_mode (string, optional): `'zeros'`, `'reflect'`,
#'   `'replicate'` or `'circular'`. Default: `'zeros'`
#' @param dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#' @param groups (int, optional): Number of blocked connections from input
#'   channels to output channels. Default: 1
#' @param bias (bool, optional): If `TRUE`, adds a learnable bias to the
#'   output. Default: `TRUE`
#'
#' @section Shape:
#' - Input: \eqn{(N, C_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C_{out}, H_{out}, W_{out})} where
#' \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \mbox{padding}[0] - \mbox{dilation}[0]
#'     \times (\mbox{kernel\_size}[0] - 1) - 1}{\mbox{stride}[0]} + 1\right\rfloor
#' }
#' \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \mbox{padding}[1] - \mbox{dilation}[1]
#'     \times (\mbox{kernel\_size}[1] - 1) - 1}{\mbox{stride}[1]} + 1\right\rfloor
#' }
#'
#' @section Attributes:
#'
#' - weight (Tensor): the learnable weights of the module of shape
#'   \eqn{(\mbox{out\_channels}, \frac{\mbox{in\_channels}}{\mbox{groups}}},
#'   \eqn{\mbox{kernel\_size[0]}, \mbox{kernel\_size[1]})}.
#'   The values of these weights are sampled from
#'   \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{in}} * \prod_{i=0}^{1}\mbox{kernel\_size}[i]}}
#' - bias (Tensor): the learnable bias of the module of shape
#'   (out_channels). If `bias` is `TRUE`,
#'   then the values of these weights are
#'   sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{in}} * \prod_{i=0}^{1}\mbox{kernel\_size}[i]}}
#'
#' @examples
#'
#' # With square kernels and equal stride
#' m <- nn_conv2d(16, 33, 3, stride = 2)
#' # non-square kernels and unequal stride and with padding
#' m <- nn_conv2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2))
#' # non-square kernels and unequal stride and with padding and dilation
#' m <- nn_conv2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2), dilation = c(3, 1))
#' input <- torch_randn(20, 16, 50, 100)
#' output <- m(input)
#' @export
nn_conv2d <- nn_module(
  "nn_conv2d",
  inherit = nn_conv_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride = 1,
                        padding = 0, dilation = 1, groups = 1,
                        bias = TRUE, padding_mode = "zeros") {
    kernel_size <- nn_util_pair(kernel_size)
    stride <- nn_util_pair(stride)

    if (is.character(padding)) {
      padding <- padding
    } else {
      padding <- nn_util_pair(padding)
    }

    dilation <- nn_util_pair(dilation)
    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      FALSE, nn_util_pair(0), groups, bias, padding_mode
    )
  },
  conv_forward_ = function(input, weight) {
    if (self$padding_mode != "zeros") {
      nnf_conv2d(
        nnf_pad(input, self$reversed_padding_repeated_twice_, mode = self$padding_mode),
        weight, self$bias, self$stride,
        nn_util_pair(0), self$dilation, self$groups
      )
    } else {
      nnf_conv2d(
        input, weight, self$bias, self$stride,
        self$padding, self$dilation, self$groups
      )
    }
  },
  forward = function(input) {
    self$conv_forward_(input, self$weight)
  }
)

#' Conv3D module
#'
#' Applies a 3D convolution over an input signal composed of several input
#' planes.
#' In the simplest case, the output value of the layer with input size \eqn{(N, C_{in}, D, H, W)}
#' and output \eqn{(N, C_{out}, D_{out}, H_{out}, W_{out})} can be precisely described as:
#'
#' \deqn{
#'   out(N_i, C_{out_j}) = bias(C_{out_j}) +
#'   \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)
#' }
#'
#' where \eqn{\star} is the valid 3D `cross-correlation` operator
#'
#' * `stride` controls the stride for the cross-correlation.
#' * `padding` controls the amount of implicit zero-paddings on both
#' sides for `padding` number of points for each dimension.
#' * `dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#' It is harder to describe, but this `link`_ has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#' `in_channels` and `out_channels` must both be divisible by
#' `groups`. For example,
#' * At groups=1, all inputs are convolved to all outputs.
#' * At groups=2, the operation becomes equivalent to having two conv
#' layers side by side, each seeing half the input channels,
#' and producing half the output channels, and both subsequently
#' concatenated.
#' * At groups= `in_channels`, each input channel is convolved with
#' its own set of filters, of size
#' \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}.
#'
#' The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:
#' - a single `int` -- in which case the same value is used for the depth, height and width dimension
#' - a `tuple` of three ints -- in which case, the first `int` is used for the depth dimension,
#' the second `int` for the height dimension and the third `int` for the width dimension
#'
#' @note
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid `cross-correlation`_,
#' and not a full `cross-correlation`_.
#' It is up to the user to add proper padding.
#'
#' @note
#' When `groups == in_channels` and `out_channels == K * in_channels`,
#' where `K` is a positive integer, this operation is also termed in
#' literature as depthwise convolution.
#' In other words, for an input of size \eqn{(N, C_{in}, D_{in}, H_{in}, W_{in})},
#' a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
#' \eqn{(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})}.
#'
#' @note
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting `torch.backends.cudnn.deterministic = TRUE`.
#' Please see the notes on :doc:`/notes/randomness` for background.
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int, tuple or str, optional): padding added to all six sides of the input. Default: 0
#' @param padding_mode (string, optional): `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`
#' @param dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#' @param groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#' @param bias (bool, optional): If `TRUE`, adds a learnable bias to the output. Default: `TRUE`
#'
#' @section Shape:
#' - Input: \eqn{(N, C_{in}, D_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C_{out}, D_{out}, H_{out}, W_{out})} where
#'  \deqn{
#'   D_{out} = \left\lfloor\frac{D_{in} + 2 \times \mbox{padding}[0] - \mbox{dilation}[0]
#'     \times (\mbox{kernel\_size}[0] - 1) - 1}{\mbox{stride}[0]} + 1\right\rfloor
#'  }
#'  \deqn{
#'   H_{out} = \left\lfloor\frac{H_{in} + 2 \times \mbox{padding}[1] - \mbox{dilation}[1]
#'     \times (\mbox{kernel\_size}[1] - 1) - 1}{\mbox{stride}[1]} + 1\right\rfloor
#'  }
#'  \deqn{
#'   W_{out} = \left\lfloor\frac{W_{in} + 2 \times \mbox{padding}[2] - \mbox{dilation}[2]
#'     \times (\mbox{kernel\_size}[2] - 1) - 1}{\mbox{stride}[2]} + 1\right\rfloor
#'  }
#'
#' @section Attributes:
#'
#' - weight (Tensor): the learnable weights of the module of shape
#' \eqn{(\mbox{out\_channels}, \frac{\mbox{in\_channels}}{\mbox{groups}},}
#' \eqn{\mbox{kernel\_size[0]}, \mbox{kernel\_size[1]}, \mbox{kernel\_size[2]})}.
#' The values of these weights are sampled from
#' \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{in}} * \prod_{i=0}^{2}\mbox{kernel\_size}[i]}}
#' - bias (Tensor):   the learnable bias of the module of shape (out_channels). If `bias` is ``True``,
#' then the values of these weights are
#' sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{in}} * \prod_{i=0}^{2}\mbox{kernel\_size}[i]}}
#'
#' @examples
#' # With square kernels and equal stride
#' m <- nn_conv3d(16, 33, 3, stride = 2)
#' # non-square kernels and unequal stride and with padding
#' m <- nn_conv3d(16, 33, c(3, 5, 2), stride = c(2, 1, 1), padding = c(4, 2, 0))
#' input <- torch_randn(20, 16, 10, 50, 100)
#' output <- m(input)
#' @export
nn_conv3d <- nn_module(
  "nn_conv3d",
  inherit = nn_conv_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride = 1,
                        padding = 0, dilation = 1, groups = 1, bias = TRUE,
                        padding_mode = "zeros") {
    kernel_size <- nn_util_triple(kernel_size)
    stride <- nn_util_triple(stride)

    if (is.character(padding)) {
      padding <- padding
    } else {
      padding <- nn_util_triple(padding)
    }

    dilation <- nn_util_triple(dilation)
    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding,
      dilation, FALSE, nn_util_triple(0), groups, bias,
      padding_mode
    )
  },
  forward = function(input) {
    if (self$padding_mode != "zeros") {
      nnf_conv3d(
        nnf_pad(input, self$reversed_padding_repeated_twice_, mode = self$padding_mode),
        self$weight, self$bias, self$stride, nn_util_triple(0),
        self$dilation, self$groups
      )
    } else {
      nnf_conv3d(
        input, self$weight, self$bias, self$stride,
        self$padding, self$dilation, self$groups
      )
    }
  }
)

nn_conv_transpose_nd <- nn_module(
  "nn_conv_transpose_nd",
  inherit = nn_conv_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, transposed, output_padding,
                        groups, bias, padding_mode) {
    if (padding_mode != "zeros") {
      value_error("Only 'zeros' padding is supported.")
    }

    super$initialize(
      in_channels, out_channels, kernel_size, stride,
      padding, dilation, transposed, output_padding,
      groups, bias, padding_mode
    )
  },
  .output_padding = function(input, output_size, stride, padding, kernel_size) {
    if (is.null(output_size)) {
      ret <- nn_util_single(self$output_padding)
    } else {
      k <- input$dim() - 2

      if (length(output_size) == (k + 2)) {
        output_size <- output_size[-c(1:2)]
      }

      if (length(output_size) != k) {
        value_error("output_size must have {k} or {k+2} elements (got {length(output_size)})")
      }

      min_sizes <- list()
      max_sizes <- list()

      for (d in seq_len(k)) {
        dim_size <- (input$size(d + 2) - 1) * stride[d] - 2 * padding[d] +
          kernel_size[d]

        min_sizes[[d]] <- dim_size
        max_sizes[[d]] <- min_sizes[[d]] + stride[d] - 1
      }

      for (i in seq_along(output_size)) {
        size <- output_size[i]
        min_size <- min_sizes[[i]]
        max_size <- max_sizes[[i]]

        if (size < min_size || size > max_size) {
          value_error(
            "requested an output of size {output_size}, but valid",
            "sizes range from {min_size} to {max_size} (for an input",
            "of size {input$size()[-c(1,2)]}"
          )
        }
      }

      res <- list()
      for (d in seq_len(k)) {
        res[[d]] <- output_size[d] - min_sizes[[d]]
      }

      ret <- res
    }
    ret
  }
)

#' ConvTranspose1D
#'
#' Applies a 1D transposed convolution operator over an input image
#' composed of several input planes.
#'
#' This module can be seen as the gradient of Conv1d with respect to its input.
#' It is also known as a fractionally-strided convolution or
#' a deconvolution (although it is not an actual deconvolution operation).
#'
#' * `stride` controls the stride for the cross-correlation.
#' * `padding` controls the amount of implicit zero-paddings on both
#' sides for `dilation * (kernel_size - 1) - padding` number of points. See note
#' below for details.
#' * `output_padding` controls the additional size added to one side
#' of the output shape. See note below for details.
#' * `dilation` controls the spacing between the kernel points; also known as the
#' à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic)
#' has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#' `in_channels` and `out_channels` must both be divisible by
#' `groups`. For example,
#'   * At groups=1, all inputs are convolved to all outputs.
#'   * At groups=2, the operation becomes equivalent to having two conv
#'     layers side by side, each seeing half the input channels,
#'     and producing half the output channels, and both subsequently
#'     concatenated.
#'   * At groups= `in_channels`, each input channel is convolved with
#'     its own set of filters (of size
#'     \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}).
#'
#' @note
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid `cross-correlation`_,
#' and not a full `cross-correlation`_.
#' It is up to the user to add proper padding.
#'
#' @note
#' The `padding` argument effectively adds `dilation * (kernel_size - 1) - padding`
#' amount of zero padding to both sizes of the input. This is set so that
#' when a `~torch.nn.Conv1d` and a `~torch.nn.ConvTranspose1d`
#' are initialized with same parameters, they are inverses of each other in
#' regard to the input and output shapes. However, when `stride > 1`,
#' `~torch.nn.Conv1d` maps multiple input shapes to the same output
#' shape. `output_padding` is provided to resolve this ambiguity by
#' effectively increasing the calculated output shape on one side. Note
#' that `output_padding` is only used to find output shape, but does
#' not actually add zero-padding to output.
#'
#' @note
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting `torch.backends.cudnn.deterministic =
#' TRUE`.
#'
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int or tuple, optional): `dilation * (kernel_size - 1) - padding` zero-padding
#'   will be added to both sides of the input. Default: 0
#' @param output_padding (int or tuple, optional): Additional size added to one side
#'   of the output shape. Default: 0
#' @param groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#' @param bias (bool, optional): If `True`, adds a learnable bias to the output. Default: `TRUE`
#' @param dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#' @inheritParams nn_conv1d
#'
#' @section Shape:
#' - Input: \eqn{(N, C_{in}, L_{in})}
#' - Output: \eqn{(N, C_{out}, L_{out})} where
#' \deqn{
#'   L_{out} = (L_{in} - 1) \times \mbox{stride} - 2 \times \mbox{padding} + \mbox{dilation}
#' \times (\mbox{kernel\_size} - 1) + \mbox{output\_padding} + 1
#' }
#'
#' @section Attributes:
#' - weight (Tensor): the learnable weights of the module of shape
#' \eqn{(\mbox{in\_channels}, \frac{\mbox{out\_channels}}{\mbox{groups}},}
#' \eqn{\mbox{kernel\_size})}.
#' The values of these weights are sampled from
#' \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{out}} * \mbox{kernel\_size}}}
#'
#' - bias (Tensor):   the learnable bias of the module of shape (out_channels).
#' If `bias` is `TRUE`, then the values of these weights are
#' sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#' \eqn{k = \frac{groups}{C_{\mbox{out}} * \mbox{kernel\_size}}}
#'
#' @examples
#' m <- nn_conv_transpose1d(32, 16, 2)
#' input <- torch_randn(10, 32, 2)
#' output <- m(input)
#' @export
nn_conv_transpose1d <- nn_module(
  "nn_conv_transpose1d",
  inherit = nn_conv_transpose_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride = 1,
                        padding = 0, output_padding = 0, groups = 1, bias = TRUE,
                        dilation = 1, padding_mode = "zeros") {
    kernel_size <- nn_util_single(kernel_size)
    stride <- nn_util_single(stride)

    if (!is.character(padding)) {
      padding <- nn_util_single(padding)
    }

    dilation <- nn_util_single(dilation)
    output_padding <- nn_util_single(output_padding)

    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      TRUE, output_padding, groups, bias, padding_mode
    )
  },
  forward = function(input, output_size = NULL) {
    if (self$padding_mode != "zeros") {
      value_error("Only `zeros` padding mode is supported for ConvTranspose1d")
    }

    output_padding <- self$.output_padding(
      input, output_size, self$stride,
      self$padding, self$kernel_size
    )


    nnf_conv_transpose1d(
      input, self$weight, self$bias, self$stride, self$padding,
      output_padding, self$groups, self$dilation
    )
  }
)

#' ConvTranpose2D module
#'
#' Applies a 2D transposed convolution operator over an input image
#' composed of several input planes.
#'
#' This module can be seen as the gradient of Conv2d with respect to its input.
#' It is also known as a fractionally-strided convolution or
#' a deconvolution (although it is not an actual deconvolution operation).
#'
#' * `stride` controls the stride for the cross-correlation.
#' * `padding` controls the amount of implicit zero-paddings on both
#' sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
#' below for details.
#' * `output_padding` controls the additional size added to one side
#' of the output shape. See note below for details.
#' * `dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#' It is harder to describe, but this `link`_ has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#' `in_channels` and `out_channels` must both be divisible by
#' `groups`. For example,
#'   * At groups=1, all inputs are convolved to all outputs.
#'   * At groups=2, the operation becomes equivalent to having two conv
#'     layers side by side, each seeing half the input channels,
#'     and producing half the output channels, and both subsequently
#'     concatenated.
#'   * At groups= `in_channels`, each input channel is convolved with
#'     its own set of filters (of size
#'     \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}).
#'
#' The parameters `kernel_size`, `stride`, `padding`, `output_padding`
#' can either be:
#' - a single ``int`` -- in which case the same value is used for the height and width dimensions
#' - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
#' and the second `int` for the width dimension
#'
#' @note
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid `cross-correlation`_,
#' and not a full `cross-correlation`. It is up to the user to add proper padding.
#'
#' @note
#' The `padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
#' amount of zero padding to both sizes of the input. This is set so that
#' when a [nn_conv2d] and a [nn_conv_transpose2d] are initialized with same
#' parameters, they are inverses of each other in
#' regard to the input and output shapes. However, when ``stride > 1``,
#' [nn_conv2d] maps multiple input shapes to the same output
#' shape. `output_padding` is provided to resolve this ambiguity by
#' effectively increasing the calculated output shape on one side. Note
#' that `output_padding` is only used to find output shape, but does
#' not actually add zero-padding to output.
#'
#' @note
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting `torch.backends.cudnn.deterministic =
#' TRUE`.
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
#'   will be added to both sides of each dimension in the input. Default: 0
#' @param output_padding (int or tuple, optional): Additional size added to one side
#'   of each dimension in the output shape. Default: 0
#' @param groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#' @param bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#' @param dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#' @inheritParams nn_conv2d
#'
#' @section Shape:
#' - Input: \eqn{(N, C_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C_{out}, H_{out}, W_{out})} where
#' \deqn{
#'   H_{out} = (H_{in} - 1) \times \mbox{stride}[0] - 2 \times \mbox{padding}[0] + \mbox{dilation}[0]
#' \times (\mbox{kernel\_size}[0] - 1) + \mbox{output\_padding}[0] + 1
#' }
#' \deqn{
#'   W_{out} = (W_{in} - 1) \times \mbox{stride}[1] - 2 \times \mbox{padding}[1] + \mbox{dilation}[1]
#' \times (\mbox{kernel\_size}[1] - 1) + \mbox{output\_padding}[1] + 1
#' }
#'
#' @section Attributes:
#' - weight (Tensor): the learnable weights of the module of shape
#'   \eqn{(\mbox{in\_channels}, \frac{\mbox{out\_channels}}{\mbox{groups}},}
#'   \eqn{\mbox{kernel\_size[0]}, \mbox{kernel\_size[1]})}.
#'   The values of these weights are sampled from
#'   \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{out}} * \prod_{i=0}^{1}\mbox{kernel\_size}[i]}}
#' - bias (Tensor):   the learnable bias of the module of shape (out_channels)
#'   If `bias` is ``True``, then the values of these weights are
#'   sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{out}} * \prod_{i=0}^{1}\mbox{kernel\_size}[i]}}
#'
#' @examples
#' # With square kernels and equal stride
#' m <- nn_conv_transpose2d(16, 33, 3, stride = 2)
#' # non-square kernels and unequal stride and with padding
#' m <- nn_conv_transpose2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2))
#' input <- torch_randn(20, 16, 50, 100)
#' output <- m(input)
#' # exact output size can be also specified as an argument
#' input <- torch_randn(1, 16, 12, 12)
#' downsample <- nn_conv2d(16, 16, 3, stride = 2, padding = 1)
#' upsample <- nn_conv_transpose2d(16, 16, 3, stride = 2, padding = 1)
#' h <- downsample(input)
#' h$size()
#' output <- upsample(h, output_size = input$size())
#' output$size()
#' @export
nn_conv_transpose2d <- nn_module(
  "nn_conv_transpose2d",
  inherit = nn_conv_transpose_nd,
  initialize = function(in_channels,
                        out_channels,
                        kernel_size,
                        stride = 1,
                        padding = 0,
                        output_padding = 0,
                        groups = 1,
                        bias = TRUE,
                        dilation = 1,
                        padding_mode = "zeros") {
    kernel_size <- nn_util_pair(kernel_size)
    stride <- nn_util_pair(stride)

    if (!is.character(padding)) {
      padding <- nn_util_pair(padding)
    }

    dilation <- nn_util_pair(dilation)
    output_padding <- nn_util_pair(output_padding)
    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      TRUE, output_padding, groups, bias, padding_mode
    )
  },
  forward = function(input, output_size = NULL) {
    if (self$padding_mode != "zeros") {
      value_error("Only `zeros` padding mode is supported for ConvTranspose2d")
    }

    output_padding <- self$.output_padding(
      input, output_size, self$stride,
      self$padding, self$kernel_size
    )

    nnf_conv_transpose2d(
      input, self$weight, self$bias, self$stride, self$padding,
      output_padding, self$groups, self$dilation
    )
  }
)

#' ConvTranpose3D module
#'
#' Applies a 3D transposed convolution operator over an input image composed of several input
#' planes.
#'
#' The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
#' and sums over the outputs from all input feature planes.
#'
#' This module can be seen as the gradient of Conv3d with respect to its input.
#' It is also known as a fractionally-strided convolution or
#' a deconvolution (although it is not an actual deconvolution operation).
#'
#' * `stride` controls the stride for the cross-correlation.
#' * `padding` controls the amount of implicit zero-paddings on both
#'   sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
#'   below for details.
#' * `output_padding` controls the additional size added to one side
#'   of the output shape. See note below for details.
#' * `dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#'   It is harder to describe, but this `link`_ has a nice visualization of what `dilation` does.
#' * `groups` controls the connections between inputs and outputs.
#'   `in_channels` and `out_channels` must both be divisible by
#'   `groups`. For example,
#'   * At groups=1, all inputs are convolved to all outputs.
#'   * At groups=2, the operation becomes equivalent to having two conv
#'     layers side by side, each seeing half the input channels,
#'     and producing half the output channels, and both subsequently
#'     concatenated.
#'   * At groups= `in_channels`, each input channel is convolved with
#'     its own set of filters (of size
#'     \eqn{\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor}).
#'
#' The parameters `kernel_size`, `stride`, `padding`, `output_padding`
#' can either be:
#'
#' - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
#' - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
#' the second `int` for the height dimension and the third `int` for the width dimension
#'
#' @note
#' Depending of the size of your kernel, several (of the last)
#' columns of the input might be lost, because it is a valid `cross-correlation`_,
#' and not a full `cross-correlation`_.
#' It is up to the user to add proper padding.
#'
#' @note
#' The `padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
#' amount of zero padding to both sizes of the input. This is set so that
#' when a `~torch.nn.Conv3d` and a `~torch.nn.ConvTranspose3d`
#' are initialized with same parameters, they are inverses of each other in
#' regard to the input and output shapes. However, when ``stride > 1``,
#' `~torch.nn.Conv3d` maps multiple input shapes to the same output
#' shape. `output_padding` is provided to resolve this ambiguity by
#' effectively increasing the calculated output shape on one side. Note
#' that `output_padding` is only used to find output shape, but does
#' not actually add zero-padding to output.
#'
#' @note
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting ``torch.backends.cudnn.deterministic =
#' TRUE``.
#'
#' @param in_channels (int): Number of channels in the input image
#' @param out_channels (int): Number of channels produced by the convolution
#' @param kernel_size (int or tuple): Size of the convolving kernel
#' @param stride (int or tuple, optional): Stride of the convolution. Default: 1
#' @param padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
#'   will be added to both sides of each dimension in the input. Default: 0
#'   output_padding (int or tuple, optional): Additional size added to one side
#'   of each dimension in the output shape. Default: 0
#' @param groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#' @param bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#' @param dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#' @inheritParams nn_conv3d
#' @inheritParams nn_conv_transpose2d
#'
#' @section Shape:
#' - Input: \eqn{(N, C_{in}, D_{in}, H_{in}, W_{in})}
#' - Output: \eqn{(N, C_{out}, D_{out}, H_{out}, W_{out})} where
#' \deqn{
#'   D_{out} = (D_{in} - 1) \times \mbox{stride}[0] - 2 \times \mbox{padding}[0] + \mbox{dilation}[0]
#' \times (\mbox{kernel\_size}[0] - 1) + \mbox{output\_padding}[0] + 1
#' }
#' \deqn{
#'   H_{out} = (H_{in} - 1) \times \mbox{stride}[1] - 2 \times \mbox{padding}[1] + \mbox{dilation}[1]
#' \times (\mbox{kernel\_size}[1] - 1) + \mbox{output\_padding}[1] + 1
#' }
#' \deqn{
#'   W_{out} = (W_{in} - 1) \times \mbox{stride}[2] - 2 \times \mbox{padding}[2] + \mbox{dilation}[2]
#' \times (\mbox{kernel\_size}[2] - 1) + \mbox{output\_padding}[2] + 1
#' }
#'
#' @section Attributes:
#' - weight (Tensor): the learnable weights of the module of shape
#'   \eqn{(\mbox{in\_channels}, \frac{\mbox{out\_channels}}{\mbox{groups}},}
#'   \eqn{\mbox{kernel\_size[0]}, \mbox{kernel\_size[1]}, \mbox{kernel\_size[2]})}.
#'   The values of these weights are sampled from
#'   \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{out}} * \prod_{i=0}^{2}\mbox{kernel\_size}[i]}}
#' - bias (Tensor):   the learnable bias of the module of shape (out_channels)
#'   If `bias` is ``True``, then the values of these weights are
#'   sampled from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{groups}{C_{\mbox{out}} * \prod_{i=0}^{2}\mbox{kernel\_size}[i]}}
#'
#' @examples
#' \dontrun{
#' # With square kernels and equal stride
#' m <- nn_conv_transpose3d(16, 33, 3, stride = 2)
#' # non-square kernels and unequal stride and with padding
#' m <- nn_conv_transpose3d(16, 33, c(3, 5, 2), stride = c(2, 1, 1), padding = c(0, 4, 2))
#' input <- torch_randn(20, 16, 10, 50, 100)
#' output <- m(input)
#' }
#' @export
nn_conv_transpose3d <- nn_module(
  "nn_conv_transpose3d",
  inherit = nn_conv_transpose_nd,
  initialize = function(in_channels, out_channels, kernel_size, stride = 1,
                        padding = 0, output_padding = 0, groups = 1, bias = TRUE,
                        dilation = 1, padding_mode = "zeros") {
    kernel_size <- nn_util_triple(kernel_size)
    stride <- nn_util_triple(stride)

    if (!is.character(padding)) {
      padding <- nn_util_triple(padding)
    }

    dilation <- nn_util_triple(dilation)
    output_padding <- nn_util_triple(output_padding)

    super$initialize(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      TRUE, output_padding, groups, bias, padding_mode
    )
  },
  forward = function(input, output_size = NULL) {
    if (self$padding_mode != "zeros") {
      value_error("Only `zeros` padding mode is supported for ConvTranspose3d")
    }

    output_padding <- self$.output_padding(
      input, output_size, self$stride,
      self$padding, self$kernel_size
    )


    nnf_conv_transpose3d(
      input, self$weight, self$bias, self$stride, self$padding,
      output_padding, self$groups, self$dilation
    )
  }
)
