#' Conv1d
#'
#' Applies a 1D convolution over an input signal composed of several input
#' planes.
#'
#' @param input input tensor of shape (minibatch, in_channels , iW)
#' @param weight filters of shape (out_channels, in_channels/groups , kW)
#' @param bias optional bias of shape (out_channels). Default: `NULL`
#' @param stride the stride of the convolving kernel. Can be a single number or
#'    a one-element tuple `(sW,)`. Default: 1
#' @param padding implicit paddings on both sides of the input. Can be a      
#'   single number or a one-element tuple `(padW,)`. Default: 0
#' @param dilation the spacing between kernel elements. Can be a single number or
#'   a one-element tuple `(dW,)`. Default: 1
#' @param groups split input into groups, `in_channels` should be divisible by
#'   the number of groups. Default: 1
#'
#' @export
nnf_conv1d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

#' Conv2d
#'
#' Applies a 2D convolution over an input image composed of several input
#' planes.
#'
#' @param input input tensor of shape (minibatch, in_channels, iH , iW)
#' @param weight filters of shape (out_channels , in_channels/groups, kH , kW)
#' @param bias optional bias tensor of shape (out_channels). Default: `NULL`
#' @param stride the stride of the convolving kernel. Can be a single number or a      
#'   tuple `(sH, sW)`. Default: 1
#' @param padding implicit paddings on both sides of the input. Can be a      
#'   single number or a tuple `(padH, padW)`. Default: 0
#' @param dilation the spacing between kernel elements. Can be a single number or
#'   a tuple `(dH, dW)`. Default: 1
#' @param groups split input into groups, `in_channels` should be divisible by the
#'   number of groups. Default: 1
#'
#' @export
nnf_conv2d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

#' Conv3d
#'
#' Applies a 3D convolution over an input image composed of several input
#' planes.
#' 
#' @param input input tensor of shape (minibatch, in_channels , iT , iH , iW)
#' @param weight filters of shape (out_channels , in_channels/groups, kT , kH , kW)
#' @param bias optional bias tensor of shape (out_channels). Default: `NULL`
#' @param stride the stride of the convolving kernel. Can be a single number or a
#'    tuple `(sT, sH, sW)`. Default: 1
#' @param padding implicit paddings on both sides of the input. Can be a      
#'    single number or a tuple `(padT, padH, padW)`. Default: 0
#' @param dilation the spacing between kernel elements. Can be a single number or 
#'    a tuple `(dT, dH, dW)`. Default: 1
#' @param groups split input into groups, `in_channels` should be divisible by
#'    the number of groups. Default: 1
#'
#' @export
nnf_conv3d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

#' Conv_transpose1d
#'
#' Applies a 1D transposed convolution operator over an input signal
#' composed of several input planes, sometimes also called "deconvolution".
#'
#' @inheritParams nnf_conv1d
#'
#' @export
nnf_conv_transpose1d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

#' Conv_transpose2d
#'
#' Applies a 2D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution".
#' 
#' @inheritParams nnf_conv2d
#' 
#' @export
nnf_conv_transpose2d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

#' Conv_transpose3d
#'
#' Applies a 3D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution"
#' 
#' @inheritParams nnf_conv3d
#' 
#' @export
nnf_conv_transpose3d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

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
nnf_fold <- function(input, output_size, kernel_size, dilation=1, padding=0, stride=1) {
  torch_col2im(self = input, output_size = nnf_util_pair(output_size), 
               kernel_size = nnf_util_pair(kernel_size), dilation = nnf_util_pair(dilation), 
               padding = nnf_util_pair(padding), stride = nnf_util_pair(stride))
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
    torch_im2col(input, nn_util_pair(kernel_size), nn_util_pair(dilation), 
                 nn_util_pair(padding), nn_util_pair(stride))
  } else {
    not_implemented_error("Input Error: Only 4D input Tensors are supported (got {input$dim()}D)")
  }
}
