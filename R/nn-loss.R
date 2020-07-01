#' @include nn.R
NULL

nn_loss <- nn_module(
  "nn_loss",
  initialize = function(reduction = "mean") {
    self$reduction <- reduction
  }
)

nn_weighted_loss <- nn_module(
  "nn_weighted_loss",
  inherit = nn_loss,
  initialize = function(weight = NULL, reduction = "mean") {
    super$initialize(reduction)
    self$register_buffer("weight", weight)
  }
)

#' Binary cross entropy loss
#' 
#' Creates a criterion that measures the Binary Cross Entropy
#' between the target and the output:
#' 
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
#' \deqn{
#'   \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
#' }
#' where \eqn{N} is the batch size. If `reduction` is not ``'none'``
#' (default `'mean'`), then
#' 
#' \deqn{
#'   \ell(x, y) = \begin{cases}
#' \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
#' \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
#' \end{cases}
#' }
#' 
#' This is used for measuring the error of a reconstruction in for example
#' an auto-encoder. Note that the targets \eqn{y} should be numbers
#' between 0 and 1.
#' 
#' Notice that if \eqn{x_n} is either 0 or 1, one of the log terms would be
#' mathematically undefined in the above loss equation. PyTorch chooses to set
#' \eqn{\log (0) = -\infty}, since \eqn{\lim_{x\to 0} \log (x) = -\infty}.
#' 
#' However, an infinite term in the loss equation is not desirable for several reasons.
#' For one, if either \eqn{y_n = 0} or \eqn{(1 - y_n) = 0}, then we would be
#' multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
#' we would also have an infinite term in our gradient, since
#' \eqn{\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty}.
#' 
#' This would make BCELoss's backward method nonlinear with respect to \eqn{x_n},
#' and using it for things like linear regression would not be straight-forward.
#' Our solution is that BCELoss clamps its log function outputs to be greater than
#' or equal to -100. This way, we can always have a finite loss value and a linear
#' backward method.
#' 
#' 
#' @param weight (Tensor, optional): a manual rescaling weight given to the loss
#'    of each batch element. If given, has to be a Tensor of size `nbatch`.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#'    ``'mean'``: the sum of the output will be divided by the number of
#'    elements in the output, ``'sum'``: the output will be summed. Note: `size_average`
#'    and `reduce` are in the process of being deprecated, and in the meantime,
#'    specifying either of those two args will override `reduction`. Default: ``'mean'``
#' 
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar. If `reduction` is ``'none'``, then \eqn{(N, *)}, same
#'   shape as input.
#' 
#' @examples
#' m <- nn_sigmoid()
#' loss <- nn_bce_loss()
#' input <- torch_randn(3, requires_grad=TRUE)
#' target <- torch_rand(3)
#' output <- loss(m(input), target)
#' output$backward()
#' 
#' @export
nn_bce_loss <- nn_module(
  "nn_bce_loss",
  inherit = nn_weighted_loss,  
  initialize = function(weight = NULL, reduction = "mean") {
    super$initialize(weight, reduction)
  },
  forward = function(input, target) {
    nnf_binary_cross_entropy(input, target, weight=self$weight, reduction=self$reduction)
  }
)