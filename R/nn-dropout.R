#' @include nn.R
NULL

nn_dropout_nd <- nn_module(
  "nn_dropout_nd",
  initialize = function(p = 0.5, inplace = FALSE) {
    if (p < 0 || p > 1) {
      value_error("dropout probability has to be between 0 and 1 but got {p}")
    }

    self$p <- p
    self$inplace <- inplace
  }
)

#' Dropout module
#'
#' During training, randomly zeroes some of the elements of the input
#' tensor with probability `p` using samples from a Bernoulli
#' distribution. Each channel will be zeroed out independently on every forward
#' call.
#'
#' This has proven to be an effective technique for regularization and
#' preventing the co-adaptation of neurons as described in the paper
#' [Improving neural networks by preventing co-adaptation of feature
#' detectors](https://arxiv.org/abs/1207.0580).
#'
#' Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
#' training. This means that during evaluation the module simply computes an
#' identity function.
#'
#' @param p probability of an element to be zeroed. Default: 0.5
#' @param inplace If set to `TRUE`, will do this operation in-place. Default: `FALSE`.
#'
#' @section Shape:
#'
#' - Input: \eqn{(*)}. Input can be of any shape
#' - Output: \eqn{(*)}. Output is of the same shape as input
#'
#' @examples
#' m <- nn_dropout(p = 0.2)
#' input <- torch_randn(20, 16)
#' output <- m(input)
#' @export
nn_dropout <- nn_module(
  "nn_dropout",
  inherit = nn_dropout_nd,
  forward = function(input) {
    nnf_dropout(input, self$p, self$training, self$inplace)
  }
)

#' Dropout2D module
#'
#' Randomly zero out entire channels (a channel is a 2D feature map,
#' e.g., the \eqn{j}-th channel of the \eqn{i}-th sample in the
#' batched input is a 2D tensor \eqn{\mbox{input}[i, j]}).
#'
#' Each channel will be zeroed out independently on every forward call with
#' probability `p` using samples from a Bernoulli distribution.
#' Usually the input comes from [nn_conv2d] modules.
#'
#' As described in the paper
#' [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280) ,
#' if adjacent pixels within feature maps are strongly correlated
#' (as is normally the case in early convolution layers) then i.i.d. dropout
#' will not regularize the activations and will otherwise just result
#' in an effective learning rate decrease.
#' In this case, [nn_dropout2d] will help promote independence between
#' feature maps and should be used instead.
#'
#' @param p (float, optional): probability of an element to be zero-ed.
#' @param inplace (bool, optional): If set to `TRUE`, will do this operation
#'   in-place
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, H, W)}
#' - Output: \eqn{(N, C, H, W)} (same shape as input)
#'
#' @examples
#' m <- nn_dropout2d(p = 0.2)
#' input <- torch_randn(20, 16, 32, 32)
#' output <- m(input)
#' @export
nn_dropout2d <- nn_module(
  "nn_dropout2d",
  inherit = nn_dropout_nd,
  forward = function(input) {
    nnf_dropout2d(input, self$p, self$training, self$inplace)
  }
)

#' Dropout3D module
#'
#' Randomly zero out entire channels (a channel is a 3D feature map,
#' e.g., the \eqn{j}-th channel of the \eqn{i}-th sample in the
#' batched input is a 3D tensor \eqn{\mbox{input}[i, j]}).
#'
#' Each channel will be zeroed out independently on every forward call with
#' probability `p` using samples from a Bernoulli distribution.
#' Usually the input comes from [nn_conv2d] modules.
#'
#' As described in the paper
#' [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280) ,
#' if adjacent pixels within feature maps are strongly correlated
#' (as is normally the case in early convolution layers) then i.i.d. dropout
#' will not regularize the activations and will otherwise just result
#' in an effective learning rate decrease.
#'
#' In this case, [nn_dropout3d] will help promote independence between
#' feature maps and should be used instead.
#'
#' @param p (float, optional): probability of an element to be zeroed.
#' @param inplace (bool, optional): If set to `TRUE`, will do this operation
#'   in-place
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, D, H, W)}
#' - Output: \eqn{(N, C, D, H, W)} (same shape as input)
#'
#' @examples
#' m <- nn_dropout3d(p = 0.2)
#' input <- torch_randn(20, 16, 4, 32, 32)
#' output <- m(input)
#' @export
nn_dropout3d <- nn_module(
  "nn_dropout3d",
  inherit = nn_dropout_nd,
  forward = function(input) {
    nnf_dropout3d(input, self$p, self$training, self$inplace)
  }
)
