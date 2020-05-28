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

nnf_conv_transpose1d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose2d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose3d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}