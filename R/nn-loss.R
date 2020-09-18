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

#' L1 loss
#' 
#' Creates a criterion that measures the mean absolute error (MAE) between each 
#' element in the input \eqn{x} and target \eqn{y}.
#' 
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described 
#' as:
#' 
#' \deqn{
#' \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_n = \left| x_n - y_n \right|,
#' }
#' 
#' where \eqn{N} is the batch size. If `reduction` is not `'none'`
#' (default `'mean'`), then:
#' 
#' \deqn{
#' \ell(x, y) =
#' \begin{array}{ll}
#' \operatorname{mean}(L), & \mbox{if reduction} = \mbox{'mean';}\\
#' \operatorname{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#' 
#' \eqn{x} and \eqn{y} are tensors of arbitrary shapes with a total
#' of \eqn{n} elements each.
#' 
#' The sum operation still operates over all the elements, and divides by \eqn{n}.
#' The division by \eqn{n} can be avoided if one sets `reduction = 'sum'`.
#' 
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#' 
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then
#'   \eqn{(N, *)}, same shape as the input
#' 
#' @examples
#' loss <- nn_l1_loss()
#' input <- torch_randn(3, 5, requires_grad=TRUE)
#' target <- torch_randn(3, 5)
#' output <- loss(input, target)
#' output$backward()
#' 
#' @export
nn_l1_loss <- nn_module(
  "nn_l1_loss",
  inherit = nn_loss,
  forward = function(input, target) {
    nnf_l1_loss(input, target, reduction = self$reduction)
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
#' where \eqn{N} is the batch size. If `reduction` is not `'none'`
#' (default `'mean'`), then
#' 
#' \deqn{
#'   \ell(x, y) = \left\{ \begin{array}{ll}
#' \mbox{mean}(L), & \mbox{if reduction} = \mbox{'mean';}\\
#' \mbox{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' \right.
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
#'    `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'    `'mean'`: the sum of the output will be divided by the number of
#'    elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'    and `reduce` are in the process of being deprecated, and in the meantime,
#'    specifying either of those two args will override `reduction`. Default: `'mean'`
#' 
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then \eqn{(N, *)}, same
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

#' CrossEntropyLoss module
#' 
#' This criterion combines [nn_log_softmax()] and `nn_nll_loss()` in one single class.
#' It is useful when training a classification problem with `C` classes.
#' 
#' If provided, the optional argument `weight` should be a 1D `Tensor`
#' assigning weight to each of the classes.
#' 
#' This is particularly useful when you have an unbalanced training set.
#' The `input` is expected to contain raw, unnormalized scores for each class.
#' `input` has to be a Tensor of size either \eqn{(minibatch, C)} or
#' \eqn{(minibatch, C, d_1, d_2, ..., d_K)}
#' with \eqn{K \geq 1} for the `K`-dimensional case (described later).
#' 
#' This criterion expects a class index in the range \eqn{[0, C-1]} as the
#' `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
#' is specified, this criterion also accepts this class index (this index may not
#' necessarily be in the class range).
#' 
#' The loss can be described as:
#' \deqn{
#'   \mbox{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
#' = -x[class] + \log\left(\sum_j \exp(x[j])\right)
#' }
#' or in the case of the `weight` argument being specified:
#' \deqn{
#'   \mbox{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)
#' }
#' 
#' The losses are averaged across observations for each minibatch.
#' Can also be used for higher dimension inputs, such as 2D images, by providing
#' an input of size \eqn{(minibatch, C, d_1, d_2, ..., d_K)} with \eqn{K \geq 1},
#' where \eqn{K} is the number of dimensions, and a target of appropriate shape
#' (see below).
#' 
#' @param weight (Tensor, optional): a manual rescaling weight given to each class.
#'   If given, has to be a Tensor of size `C`
#' @param ignore_index (int, optional): Specifies a target value that is ignored
#'   and does not contribute to the input gradient. When `size_average` is
#'   `TRUE`, the loss is averaged over non-ignored targets.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'   
#' @section Shape:
#' - Input: \eqn{(N, C)} where `C = number of classes`, or
#' \eqn{(N, C, d_1, d_2, ..., d_K)} with \eqn{K \geq 1}
#' in the case of `K`-dimensional loss.
#' - Target: \eqn{(N)} where each value is \eqn{0 \leq \mbox{targets}[i] \leq C-1}, or
#' \eqn{(N, d_1, d_2, ..., d_K)} with \eqn{K \geq 1} in the case of
#' K-dimensional loss.
#' - Output: scalar.
#' If `reduction` is `'none'`, then the same size as the target:
#'   \eqn{(N)}, or
#' \eqn{(N, d_1, d_2, ..., d_K)} with \eqn{K \geq 1} in the case
#' of K-dimensional loss.
#' 
#' @examples
#' loss <- nn_cross_entropy_loss()
#' input <- torch_randn(3, 5, requires_grad=TRUE)
#' target <- torch_randint(low = 1, high = 5, size = 3, dtype = torch_long())
#' output <- loss(input, target)
#' output$backward()
#' 
#' @export
nn_cross_entropy_loss <- nn_module(
  "nn_crossentropy_loss",
  inherit = nn_weighted_loss,
  initialize = function(weight = NULL, ignore_index = -100, reduction = "mean") {
    self$ignore_index <- ignore_index
    super$initialize(weight, reduction)
  },
  forward = function(input, target) {
    nnf_cross_entropy(input, target, weight = self$weight, 
                      ignore_index = self$ignore_index, reduction = self$reduction)
  }
)
