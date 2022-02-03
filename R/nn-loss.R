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
#' \mbox{mean}(L), & \mbox{if reduction} = \mbox{'mean';}\\
#' \mbox{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
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
#' input <- torch_randn(3, 5, requires_grad = TRUE)
#' target <- torch_randn(3, 5)
#' output <- loss(input, target)
#' output$backward()
#' @export
nn_l1_loss <- nn_module(
  "nn_l1_loss",
  inherit = nn_loss,
  forward = function(input, target) {
    nnf_l1_loss(input, target, reduction = self$reduction)
  }
)

#' Nll loss
#'
#' The negative log likelihood loss. It is useful to train a classification
#' problem with `C` classes.
#'
#' If provided, the optional argument `weight` should be a 1D Tensor assigning
#' weight to each of the classes. This is particularly useful when you have an
#' unbalanced training set.
#'
#' The `input` given through a forward call is expected to contain
#' log-probabilities of each class. `input` has to be a Tensor of size either
#' \eqn{(minibatch, C)} or \eqn{(minibatch, C, d_1, d_2, ..., d_K)}
#' with \eqn{K \geq 1} for the `K`-dimensional case (described later).
#'
#' Obtaining log-probabilities in a neural network is easily achieved by
#' adding a  `LogSoftmax`  layer in the last layer of your network.
#'
#' You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
#' layer.
#'
#' The `target` that this loss expects should be a class index in the range \eqn{[0, C-1]}
#' where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
#' this class index (this index may not necessarily be in the class range).
#'
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
#'
#' \deqn{
#' \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_n = - w_{y_n} x_{n,y_n}, \quad
#' w_{c} = \mbox{weight}[c] \cdot \mbox{1}\{c \not= \mbox{ignore\_index}\},
#' }
#'
#' where \eqn{x} is the input, \eqn{y} is the target, \eqn{w} is the weight, and
#' \eqn{N} is the batch size. If `reduction` is not `'none'`
#' (default `'mean'`), then
#'
#' \deqn{
#' \ell(x, y) = \begin{array}{ll}
#' \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
#'   \mbox{if reduction} = \mbox{'mean';}\\
#' \sum_{n=1}^N l_n,  &
#'   \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#'
#' Can also be used for higher dimension inputs, such as 2D images, by providing
#' an input of size \eqn{(minibatch, C, d_1, d_2, ..., d_K)} with \eqn{K \geq 1},
#' where \eqn{K} is the number of dimensions, and a target of appropriate shape
#' (see below). In the case of images, it computes NLL loss per-pixel.
#'
#'
#' @param weight (Tensor, optional): a manual rescaling weight given to each
#'   class. If given, it has to be a Tensor of size `C`. Otherwise, it is
#'   treated as if having all ones.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will
#'   be applied, `'mean'`: the weighted mean of the output is taken,
#'   `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in
#'   the meantime, specifying either of those two args will override
#'   `reduction`. Default: `'mean'`
#' @param ignore_index (int, optional): Specifies a target value that is ignored
#'   and does not contribute to the input gradient.
#'
#' @section Shape:
#' - Input: \eqn{(N, C)} where `C = number of classes`, or
#'   \eqn{(N, C, d_1, d_2, ..., d_K)} with \eqn{K \geq 1}
#'   in the case of `K`-dimensional loss.
#' - Target: \eqn{(N)} where each value is \eqn{0 \leq \mbox{targets}[i] \leq C-1}, or
#'   \eqn{(N, d_1, d_2, ..., d_K)} with \eqn{K \geq 1} in the case of
#'   K-dimensional loss.
#' - Output: scalar.
#'
#' If `reduction` is `'none'`, then the same size as the target: \eqn{(N)}, or
#' \eqn{(N, d_1, d_2, ..., d_K)} with \eqn{K \geq 1} in the case
#' of K-dimensional loss.
#'
#' @examples
#' m <- nn_log_softmax(dim = 2)
#' loss <- nn_nll_loss()
#' # input is of size N x C = 3 x 5
#' input <- torch_randn(3, 5, requires_grad = TRUE)
#' # each element in target has to have 0 <= value < C
#' target <- torch_tensor(c(2, 1, 5), dtype = torch_long())
#' output <- loss(m(input), target)
#' output$backward()
#'
#' # 2D loss example (used, for example, with image inputs)
#' N <- 5
#' C <- 4
#' loss <- nn_nll_loss()
#' # input is of size N x C x height x width
#' data <- torch_randn(N, 16, 10, 10)
#' conv <- nn_conv2d(16, C, c(3, 3))
#' m <- nn_log_softmax(dim = 1)
#' # each element in target has to have 0 <= value < C
#' target <- torch_empty(N, 8, 8, dtype = torch_long())$random_(1, C)
#' output <- loss(m(conv(data)), target)
#' output$backward()
#' @export
nn_nll_loss <- nn_module(
  "nn_nll_loss",
  inherit = nn_weighted_loss,
  initialize = function(weight = NULL, ignore_index = -100, reduction = "mean") {
    super$initialize(weight, reduction)
    self$ignore_index <- ignore_index
  },
  forward = function(input, target) {
    nnf_nll_loss(input, target,
      weight = self$weight,
      ignore_index = self$ignore_index, reduction = self$reduction
    )
  }
)

#' Poisson NLL loss
#'
#' Negative log likelihood loss with Poisson distribution of target.
#' The loss can be described as:
#'
#' \deqn{
#' \mbox{target} \sim \mathrm{Poisson}(\mbox{input})
#' \mbox{loss}(\mbox{input}, \mbox{target}) = \mbox{input} - \mbox{target} * \log(\mbox{input})
#' + \log(\mbox{target!})
#' }
#'
#' The last term can be omitted or approximated with Stirling formula. The
#' approximation is used for target values more than 1. For targets less or
#' equal to 1 zeros are added to the loss.
#'
#' @param log_input (bool, optional): if `TRUE` the loss is computed as
#'   \eqn{\exp(\mbox{input}) - \mbox{target}*\mbox{input}}, if `FALSE` the loss is
#'   \eqn{\mbox{input} - \mbox{target}*\log(\mbox{input}+\mbox{eps})}.
#' @param full (bool, optional): whether to compute full loss, i. e. to add the
#'   Stirling approximation term
#'   \eqn{\mbox{target}*\log(\mbox{target}) - \mbox{target} + 0.5 * \log(2\pi\mbox{target})}.
#' @param eps (float, optional): Small value to avoid evaluation of \eqn{\log(0)} when
#'   `log_input = FALSE`. Default: 1e-8
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @examples
#' loss <- nn_poisson_nll_loss()
#' log_input <- torch_randn(5, 2, requires_grad = TRUE)
#' target <- torch_randn(5, 2)
#' output <- loss(log_input, target)
#' output$backward()
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar by default. If `reduction` is `'none'`, then \eqn{(N, *)},
#'   the same shape as the input
#'
#' @export
nn_poisson_nll_loss <- nn_module(
  "nn_poisson_nll_loss",
  inherit = nn_loss,
  initialize = function(log_input = TRUE, full = FALSE, eps = 1e-8,
                        reduction = "mean") {
    super$initialize(reduction = reduction)
    self$log_input <- log_input
    self$full <- full
    self$eps <- eps
  },
  forward = function(log_input, target) {
    nnf_poisson_nll_loss(log_input, target,
      log_input = self$log_input,
      full = self$full, eps = self$eps,
      reduction = self$reduction
    )
  }
)

#' Kullback-Leibler divergence loss
#'
#' The Kullback-Leibler divergence loss measure
#' [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)
#' is a useful distance measure for continuous distributions and is often
#' useful when performing direct regression over the space of (discretely sampled)
#' continuous output distributions.
#'
#' As with [nn_nll_loss()], the `input` given is expected to contain
#' *log-probabilities* and is not restricted to a 2D Tensor.
#'
#' The targets are interpreted as *probabilities* by default, but could be considered
#' as *log-probabilities* with `log_target` set to `TRUE`.
#'
#' This criterion expects a `target` `Tensor` of the same size as the
#' `input` `Tensor`.
#'
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described
#' as:
#'
#' \deqn{
#'   l(x,y) = L = \{ l_1,\dots,l_N \}, \quad
#' l_n = y_n \cdot \left( \log y_n - x_n \right)
#' }
#'
#' where the index \eqn{N} spans all dimensions of `input` and \eqn{L} has the same
#' shape as `input`. If `reduction` is not `'none'` (default `'mean'`), then:
#'
#' \deqn{
#'   \ell(x, y) = \begin{array}{ll}
#' \mbox{mean}(L), & \mbox{if reduction} = \mbox{'mean';} \\
#' \mbox{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#'
#' In default `reduction` mode `'mean'`, the losses are averaged for each minibatch
#' over observations **as well as** over dimensions. `'batchmean'` mode gives the
#' correct KL divergence where losses are averaged over batch dimension only.
#' `'mean'` mode's behavior will be changed to the same as `'batchmean'` in the next
#' major release.
#'
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'  `'none'` | `'batchmean'` | `'sum'` | `'mean'`.
#'  `'none'`: no reduction will be applied.
#'  `'batchmean'`: the sum of the output will be divided by batchsize.
#'  `'sum'`: the output will be summed.
#'  `'mean'`: the output will be divided by the number of elements in the output.
#'  Default: `'mean'`
#'
#' @note
#' `reduction` = `'mean'` doesn't return the true kl divergence value,
#' please use `reduction` = `'batchmean'` which aligns with KL math
#' definition.
#' In the next major release, `'mean'` will be changed to be the same as
#' `'batchmean'`.
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar by default. If `reduction` is `'none'`, then \eqn{(N, *)},
#'   the same shape as the input
#'
#' @export
nn_kl_div_loss <- nn_module(
  "nn_kl_div_loss",
  inherit = nn_loss,
  initialize = function(reduction = "mean") {
    super$initialize(reduction = reduction)
  },
  forward = function(input, target) {
    nnf_kl_div(input, target, reduction = self$reduction)
  }
)

#' MSE loss
#'
#' Creates a criterion that measures the mean squared error (squared L2 norm) between
#' each element in the input \eqn{x} and target \eqn{y}.
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described
#' as:
#'
#' \deqn{
#'   \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_n = \left( x_n - y_n \right)^2,
#' }
#'
#' where \eqn{N} is the batch size. If `reduction` is not `'none'`
#' (default `'mean'`), then:
#'
#' \deqn{
#'   \ell(x, y) =
#'   \begin{array}{ll}
#' \mbox{mean}(L), &  \mbox{if reduction} = \mbox{'mean';}\\
#' \mbox{sum}(L),  &  \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#'
#' \eqn{x} and \eqn{y} are tensors of arbitrary shapes with a total
#' of \eqn{n} elements each.
#'
#' The mean operation still operates over all the elements, and divides by \eqn{n}.
#' The division by \eqn{n} can be avoided if one sets `reduction = 'sum'`.
#'
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'  `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'  `'mean'`: the sum of the output will be divided by the number of
#'  elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'  and `reduce` are in the process of being deprecated, and in the meantime,
#'  specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' loss <- nn_mse_loss()
#' input <- torch_randn(3, 5, requires_grad = TRUE)
#' target <- torch_randn(3, 5)
#' output <- loss(input, target)
#' output$backward()
#' @export
nn_mse_loss <- nn_module(
  "nn_mse_loss",
  inherit = nn_loss,
  initialize = function(reduction = "mean") {
    super$initialize(reduction = reduction)
  },
  forward = function(input, target) {
    nnf_mse_loss(input, target, reduction = self$reduction)
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
#' input <- torch_randn(3, requires_grad = TRUE)
#' target <- torch_rand(3)
#' output <- loss(m(input), target)
#' output$backward()
#' @export
nn_bce_loss <- nn_module(
  "nn_bce_loss",
  inherit = nn_weighted_loss,
  initialize = function(weight = NULL, reduction = "mean") {
    super$initialize(weight, reduction)
  },
  forward = function(input, target) {
    nnf_binary_cross_entropy(input, target, weight = self$weight, reduction = self$reduction)
  }
)

#' BCE with logits loss
#'
#' This loss combines a `Sigmoid` layer and the `BCELoss` in one single
#' class. This version is more numerically stable than using a plain `Sigmoid`
#' followed by a `BCELoss` as, by combining the operations into one layer,
#' we take advantage of the log-sum-exp trick for numerical stability.
#'
#' The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
#'
#' \deqn{
#'   \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
#'                    + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],
#' }
#'
#' where \eqn{N} is the batch size. If `reduction` is not `'none'`
#' (default `'mean'`), then
#'
#' \deqn{
#'   \ell(x, y) = \begin{array}{ll}
#' \mbox{mean}(L), & \mbox{if reduction} = \mbox{'mean';}\\
#' \mbox{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#'
#' This is used for measuring the error of a reconstruction in for example
#' an auto-encoder. Note that the targets `t[i]` should be numbers
#' between 0 and 1.
#' It's possible to trade off recall and precision by adding weights to positive examples.
#' In the case of multi-label classification the loss can be described as:
#'
#' \deqn{
#' \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
#' l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
#' + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],
#' }
#' where \eqn{c} is the class number (\eqn{c > 1} for multi-label binary
#' classification,
#'
#' \eqn{c = 1} for single-label binary classification),
#' \eqn{n} is the number of the sample in the batch and
#' \eqn{p_c} is the weight of the positive answer for the class \eqn{c}.
#' \eqn{p_c > 1} increases the recall, \eqn{p_c < 1} increases the precision.
#' For example, if a dataset contains 100 positive and 300 negative examples of a single class,
#' then `pos_weight` for the class should be equal to \eqn{\frac{300}{100}=3}.
#' The loss would act as if the dataset contains \eqn{3\times 100=300} positive examples.
#'
#' @param weight (Tensor, optional): a manual rescaling weight given to the loss
#'   of each batch element. If given, has to be a Tensor of size `nbatch`.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#' @param pos_weight (Tensor, optional): a weight of positive examples.
#'   Must be a vector with length equal to the number of classes.
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then \eqn{(N, *)}, same
#'   shape as input.
#'
#' @examples
#' loss <- nn_bce_with_logits_loss()
#' input <- torch_randn(3, requires_grad = TRUE)
#' target <- torch_empty(3)$random_(1, 2)
#' output <- loss(input, target)
#' output$backward()
#'
#' target <- torch_ones(10, 64, dtype = torch_float32()) # 64 classes, batch size = 10
#' output <- torch_full(c(10, 64), 1.5) # A prediction (logit)
#' pos_weight <- torch_ones(64) # All weights are equal to 1
#' criterion <- nn_bce_with_logits_loss(pos_weight = pos_weight)
#' criterion(output, target) # -log(sigmoid(1.5))
#' @export
nn_bce_with_logits_loss <- nn_module(
  "nn_bce_with_logits_loss",
  inherit = nn_loss,
  initialize = function(weight = NULL, reduction = "mean", pos_weight = NULL) {
    super$initialize(reduction = reduction)
    self$register_buffer("weight", weight)
    self$register_buffer("pos_weight", pos_weight)
  },
  forward = function(input, target) {
    nnf_binary_cross_entropy_with_logits(
      input, target, self$weight,
      pos_weight = self$pos_weight,
      reduction = self$reduction
    )
  }
)

#' Hinge embedding loss
#'
#' Measures the loss given an input tensor \eqn{x} and a labels tensor \eqn{y}
#' (containing 1 or -1).
#'
#' This is usually used for measuring whether two inputs are similar or
#' dissimilar, e.g. using the L1 pairwise distance as \eqn{x}, and is typically
#' used for learning nonlinear embeddings or semi-supervised learning.
#' The loss function for \eqn{n}-th sample in the mini-batch is
#'
#' \deqn{
#'   l_n = \begin{array}{ll}
#' x_n, & \mbox{if}\; y_n = 1,\\
#' \max \{0, \Delta - x_n\}, & \mbox{if}\; y_n = -1,
#' \end{array}
#' }
#'
#' and the total loss functions is
#'
#' \deqn{
#'   \ell(x, y) = \begin{array}{ll}
#' \mbox{mean}(L), & \mbox{if reduction} = \mbox{'mean';}\\
#' \mbox{sum}(L),  & \mbox{if reduction} = \mbox{'sum'.}
#' \end{array}
#' }
#'
#' where \eqn{L = \{l_1,\dots,l_N\}^\top}.
#'
#' @param margin (float, optional): Has a default value of `1`.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'  `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'  `'mean'`: the sum of the output will be divided by the number of
#'  elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'  and `reduce` are in the process of being deprecated, and in the meantime,
#'  specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(*)} where \eqn{*} means, any number of dimensions. The sum operation
#'   operates over all the elements.
#' - Target: \eqn{(*)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then same shape as the input
#'
#' @export
nn_hinge_embedding_loss <- nn_module(
  "nn_hinge_embedding_loss",
  inherit = nn_loss,
  initialize = function(margin = 1.0, reduction = "mean") {
    super$initialize(reduction = reduction)
    self$margin <- margin
  },
  forward = function(input, target) {
    nnf_hinge_embedding_loss(input, target,
      margin = self$margin,
      reduction = self$reduction
    )
  }
)

#' Multilabel margin loss
#'
#' Creates a criterion that optimizes a multi-class multi-classification
#' hinge loss (margin-based loss) between input \eqn{x} (a 2D mini-batch `Tensor`)
#' and output \eqn{y} (which is a 2D `Tensor` of target class indices).
#' For each sample in the mini-batch:
#'
#' \deqn{
#'   \mbox{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\mbox{x.size}(0)}
#' }
#'
#' where \eqn{x \in \left\{0, \; \cdots , \; \mbox{x.size}(0) - 1\right\}}, \
#' \eqn{y \in \left\{0, \; \cdots , \; \mbox{y.size}(0) - 1\right\}}, \
#' \eqn{0 \leq y[j] \leq \mbox{x.size}(0)-1}, \
#' and \eqn{i \neq y[j]} for all \eqn{i} and \eqn{j}.
#' \eqn{y} and \eqn{x} must have the same size.
#'
#' The criterion only considers a contiguous block of non-negative targets that
#' starts at the front.
#' This allows for different samples to have variable amounts of target classes.
#'
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'  `'mean'`: the sum of the output will be divided by the number of
#'  elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'  and `reduce` are in the process of being deprecated, and in the meantime,
#'  specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(C)} or \eqn{(N, C)} where `N` is the batch size and `C`
#'   is the number of classes.
#' - Target: \eqn{(C)} or \eqn{(N, C)}, label targets padded by -1 ensuring same shape as the input.
#' - Output: scalar. If `reduction` is `'none'`, then \eqn{(N)}.
#'
#' @examples
#' loss <- nn_multilabel_margin_loss()
#' x <- torch_tensor(c(0.1, 0.2, 0.4, 0.8))$view(c(1, 4))
#' # for target y, only consider labels 4 and 1, not after label -1
#' y <- torch_tensor(c(4, 1, -1, 2), dtype = torch_long())$view(c(1, 4))
#' loss(x, y)
#' @export
nn_multilabel_margin_loss <- nn_module(
  "nn_multilabel_margin_loss",
  inherit = nn_loss,
  initialize = function(reduction = "mean") {
    super$initialize(reduction = reduction)
  },
  forward = function(input, target) {
    nnf_multilabel_margin_loss(input, target, reduction = self$reduction)
  }
)

#' Smooth L1 loss
#'
#' Creates a criterion that uses a squared term if the absolute
#' element-wise error falls below 1 and an L1 term otherwise.
#' It is less sensitive to outliers than the `MSELoss` and in some cases
#' prevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).
#' Also known as the Huber loss:
#'
#' \deqn{
#'   \mbox{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}
#' }
#'
#' where \eqn{z_{i}} is given by:
#'
#' \deqn{
#'   z_{i} =
#'   \begin{array}{ll}
#' 0.5 (x_i - y_i)^2, & \mbox{if } |x_i - y_i| < 1 \\
#' |x_i - y_i| - 0.5, & \mbox{otherwise }
#' \end{array}
#' }
#'
#' \eqn{x} and \eqn{y} arbitrary shapes with a total of \eqn{n} elements each
#' the sum operation still operates over all the elements, and divides by \eqn{n}.
#' The division by \eqn{n} can be avoided if sets `reduction = 'sum'`.
#'
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'  `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'  `'mean'`: the sum of the output will be divided by the number of
#'  elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'  and `reduce` are in the process of being deprecated, and in the meantime,
#'  specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(N, *)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then
#'   \eqn{(N, *)}, same shape as the input
#'
#' @export
nn_smooth_l1_loss <- nn_module(
  "nn_smooth_l1_loss",
  inherit = nn_loss,
  initialize = function(reduction = "mean") {
    super$initialize(reduction = reduction)
  },
  forward = function(input, target) {
    nnf_smooth_l1_loss(input, target, reduction = self$reduction)
  }
)

#' Soft margin loss
#'
#' Creates a criterion that optimizes a two-class classification
#' logistic loss between input tensor \eqn{x} and target tensor \eqn{y}
#' (containing 1 or -1).
#'
#' \deqn{
#'   \mbox{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\mbox{x.nelement}()}
#' }
#'
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(*)} where \eqn{*} means, any number of additional
#'   dimensions
#' - Target: \eqn{(*)}, same shape as the input
#' - Output: scalar. If `reduction` is `'none'`, then same shape as the input
#'
#' @export
nn_soft_margin_loss <- nn_module(
  "nn_soft_margin_loss",
  inherit = nn_loss,
  initialize = function(reduction = "mean") {
    super$initialize(reduction = reduction)
  },
  forward = function(input, target) {
    nnf_soft_margin_loss(input, target, reduction = self$reduction)
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
#' input <- torch_randn(3, 5, requires_grad = TRUE)
#' target <- torch_randint(low = 1, high = 5, size = 3, dtype = torch_long())
#' output <- loss(input, target)
#' output$backward()
#' @export
nn_cross_entropy_loss <- nn_module(
  "nn_crossentropy_loss",
  inherit = nn_weighted_loss,
  initialize = function(weight = NULL, ignore_index = -100, reduction = "mean") {
    self$ignore_index <- ignore_index
    super$initialize(weight, reduction)
  },
  forward = function(input, target) {
    nnf_cross_entropy(input, target,
      weight = self$weight,
      ignore_index = self$ignore_index, reduction = self$reduction
    )
  }
)

#' Multi label soft margin loss
#'
#' Creates a criterion that optimizes a multi-label one-versus-all
#' loss based on max-entropy, between input \eqn{x} and target \eqn{y} of size
#' \eqn{(N, C)}.
#'
#' For each sample in the minibatch:
#'
#' \deqn{
#'   loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
#' + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)
#' }
#'
#' where \eqn{i \in \left\{0, \; \cdots , \; \mbox{x.nElement}() - 1\right\}},
#' \eqn{y[i] \in \left\{0, \; 1\right\}}.
#'
#' @param weight (Tensor, optional): a manual rescaling weight given to each
#'   class. If given, it has to be a Tensor of size `C`. Otherwise, it is
#'   treated as if having all ones.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(N, C)} where `N` is the batch size and `C` is the number of classes.
#' - Target: \eqn{(N, C)}, label targets padded by -1 ensuring same shape as the input.
#' - Output: scalar. If `reduction` is `'none'`, then \eqn{(N)}.
#'
#' @export
nn_multilabel_soft_margin_loss <- nn_module(
  "nn_multilabel_soft_margin_loss",
  inherit = nn_weighted_loss,
  initialize = function(weight = NULL, reduction = "mean") {
    super$initialize(weight = weight, reduction = reduction)
  },
  forward = function(input, target) {
    nnf_multilabel_soft_margin_loss(input, target,
      weight = self$weight,
      reduction = self$reduction
    )
  }
)

#' Cosine embedding loss
#'
#' Creates a criterion that measures the loss given input tensors
#' \eqn{x_1}, \eqn{x_2} and a `Tensor` label \eqn{y} with values 1 or -1.
#' This is used for measuring whether two inputs are similar or dissimilar,
#' using the cosine distance, and is typically used for learning nonlinear
#' embeddings or semi-supervised learning.
#' The loss function for each sample is:
#'
#' \deqn{
#'   \mbox{loss}(x, y) =
#'   \begin{array}{ll}
#' 1 - \cos(x_1, x_2), & \mbox{if } y = 1 \\
#' \max(0, \cos(x_1, x_2) - \mbox{margin}), & \mbox{if } y = -1
#' \end{array}
#' }
#'
#' @param margin (float, optional): Should be a number from \eqn{-1} to \eqn{1},
#'   \eqn{0} to \eqn{0.5} is suggested. If `margin` is missing, the
#'   default value is \eqn{0}.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @export
nn_cosine_embedding_loss <- nn_module(
  "nn_cosine_embedding_loss",
  inherit = nn_loss,
  initialize = function(margin = 0, reduction = "mean") {
    super$initialize(reduction = reduction)
    self$margin <- margin
  },
  forward = function(input1, input2, target) {
    nnf_cosine_embedding_loss(input1, input2, target,
      margin = self$margin,
      reduction = self$reduction
    )
  }
)

#' Margin ranking loss
#'
#' Creates a criterion that measures the loss given
#' inputs \eqn{x1}, \eqn{x2}, two 1D mini-batch `Tensors`,
#' and a label 1D mini-batch tensor \eqn{y} (containing 1 or -1).
#' If \eqn{y = 1} then it assumed the first input should be ranked higher
#' (have a larger value) than the second input, and vice-versa for \eqn{y = -1}.
#'
#' The loss function for each pair of samples in the mini-batch is:
#'
#' \deqn{
#'   \mbox{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \mbox{margin})
#' }
#'
#'
#' @param margin (float, optional): Has a default value of \eqn{0}.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input1: \eqn{(N)} where `N` is the batch size.
#' - Input2: \eqn{(N)}, same shape as the Input1.
#' - Target: \eqn{(N)}, same shape as the inputs.
#' - Output: scalar. If `reduction` is `'none'`, then \eqn{(N)}.
#'
#' @examples
#' loss <- nn_margin_ranking_loss()
#' input1 <- torch_randn(3, requires_grad = TRUE)
#' input2 <- torch_randn(3, requires_grad = TRUE)
#' target <- torch_randn(3)$sign()
#' output <- loss(input1, input2, target)
#' output$backward()
#' @export
nn_margin_ranking_loss <- nn_module(
  "nn_margin_ranking_loss",
  inherit = nn_loss,
  initialize = function(margin = 0, reduction = "mean") {
    super$initialize(reduction = reduction)
    self$margin <- margin
  },
  forward = function(input1, input2, target) {
    nnf_margin_ranking_loss(input1, input2, target,
      margin = self$margin,
      reduction = self$reduction
    )
  }
)

#' Multi margin loss
#'
#' Creates a criterion that optimizes a multi-class classification hinge
#' loss (margin-based loss) between input \eqn{x} (a 2D mini-batch `Tensor`) and
#' output \eqn{y} (which is a 1D tensor of target class indices,
#' \eqn{0 \leq y \leq \mbox{x.size}(1)-1}):
#'
#' For each mini-batch sample, the loss in terms of the 1D input \eqn{x} and scalar
#' output \eqn{y} is:
#' \deqn{
#'   \mbox{loss}(x, y) = \frac{\sum_i \max(0, \mbox{margin} - x[y] + x[i]))^p}{\mbox{x.size}(0)}
#' }
#'
#' where \eqn{x \in \left\{0, \; \cdots , \; \mbox{x.size}(0) - 1\right\}}
#' and \eqn{i \neq y}.
#'
#' Optionally, you can give non-equal weighting on the classes by passing
#' a 1D `weight` tensor into the constructor.
#' The loss function then becomes:
#'
#' \deqn{
#'   \mbox{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\mbox{margin} - x[y] + x[i]))^p)}{\mbox{x.size}(0)}
#' }
#'
#' @param p (int, optional): Has a default value of \eqn{1}. \eqn{1} and \eqn{2}
#'   are the only supported values.
#' @param margin (float, optional): Has a default value of \eqn{1}.
#' @param weight (Tensor, optional): a manual rescaling weight given to each
#'   class. If given, it has to be a Tensor of size `C`. Otherwise, it is
#'   treated as if having all ones.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @export
nn_multi_margin_loss <- nn_module(
  "nn_multi_margin_loss",
  inherit = nn_weighted_loss,
  initialize = function(p = 1, margin = 1, weight = NULL, reduction = "mean") {
    super$initialize(weight = weight, reduction = reduction)
    if (p != 1 && p != 2) {
      value_error("only p == 1 or p == 2 are supported.")
    }
    if (!is.null(weight) && weight$dim() != 1) {
      value_error("weight must be NULL or 1-dimensional")
    }
    self$p <- p
    self$margin <- margin
  },
  forward = function(input, target) {
    nnf_multi_margin_loss(input, target,
      p = self$p, margin = self$margin,
      weight = self$weight, reduction = self$reduction
    )
  }
)

#' Triplet margin loss
#'
#' Creates a criterion that measures the triplet loss given an input
#' tensors \eqn{x1}, \eqn{x2}, \eqn{x3} and a margin with a value greater than \eqn{0}.
#' This is used for measuring a relative similarity between samples. A triplet
#' is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
#' examples` respectively). The shapes of all input tensors should be
#' \eqn{(N, D)}.
#'
#' The distance swap is described in detail in the paper
#' [Learning shallow convolutional feature descriptors with triplet losses](http://www.bmva.org/bmvc/2016/papers/paper119/index.html) by
#' V. Balntas, E. Riba et al.
#'
#' The loss function for each sample in the mini-batch is:
#'
#' \deqn{
#'   L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}
#' }
#'
#' where
#'
#' \deqn{
#'   d(x_i, y_i) = | {\bf x}_i - {\bf y}_i |_p
#' }
#'
#' See also [nn_triplet_margin_with_distance_loss()], which computes the
#' triplet margin loss for input tensors using a custom distance function.
#'
#' @param margin (float, optional): Default: \eqn{1}.
#' @param p (int, optional): The norm degree for pairwise distance. Default: \eqn{2}.
#' @param swap (bool, optional): The distance swap is described in detail in the paper
#'   [Learning shallow convolutional feature descriptors with triplet losses](http://www.bmva.org/bmvc/2016/papers/paper119/index.html) by
#'   V. Balntas, E. Riba et al. Default: `FALSE`.
#' @param eps constant to avoid NaN's
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Note: `size_average`
#'   and `reduce` are in the process of being deprecated, and in the meantime,
#'   specifying either of those two args will override `reduction`. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(N, D)} where \eqn{D} is the vector dimension.
#' - Output: A Tensor of shape \eqn{(N)} if `reduction` is `'none'`, or a scalar
#'   otherwise.
#'
#' @examples
#' triplet_loss <- nn_triplet_margin_loss(margin = 1, p = 2)
#' anchor <- torch_randn(100, 128, requires_grad = TRUE)
#' positive <- torch_randn(100, 128, requires_grad = TRUE)
#' negative <- torch_randn(100, 128, requires_grad = TRUE)
#' output <- triplet_loss(anchor, positive, negative)
#' output$backward()
#' @export
nn_triplet_margin_loss <- nn_module(
  "nn_triplet_margin_loss",
  inherit = nn_loss,
  initialize = function(margin = 1, p = 2, eps = 1e-6, swap = FALSE,
                        reduction = "mean") {
    super$initialize(reduction = reduction)
    self$margin <- margin
    self$p <- p
    self$eps <- eps
    self$swap <- swap
  },
  forward = function(anchor, positive, negative) {
    nnf_triplet_margin_loss(anchor, positive, negative,
      margin = self$margin,
      p = self$p, eps = self$eps, swap = self$swap
    )
  }
)

#' Triplet margin with distance loss
#'
#' Creates a criterion that measures the triplet loss given input
#' tensors \eqn{a}, \eqn{p}, and \eqn{n} (representing anchor,
#' positive, and negative examples, respectively), and a nonnegative,
#' real-valued function ("distance function") used to compute the relationship
#' between the anchor and positive example ("positive distance") and the
#' anchor and negative example ("negative distance").
#'
#' The unreduced loss (i.e., with `reduction` set to `'none'`)
#' can be described as:
#'
#' \deqn{
#'   \ell(a, p, n) = L = \{l_1,\dots,l_N\}^\top, \quad
#' l_i = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}
#' }
#'
#' where \eqn{N} is the batch size; \eqn{d} is a nonnegative, real-valued function
#' quantifying the closeness of two tensors, referred to as the `distance_function`;
#' and \eqn{margin} is a non-negative margin representing the minimum difference
#' between the positive and negative distances that is required for the loss to
#' be 0.  The input tensors have \eqn{N} elements each and can be of any shape
#' that the distance function can handle.
#' If `reduction` is not `'none'`
#' (default `'mean'`), then:
#'
#' \deqn{
#' \ell(x, y) =
#' \begin{array}{ll}
#' \mbox{mean}(L), &  \mbox{if reduction} = \mbox{`mean';}\\
#'             \mbox{sum}(L),  &  \mbox{if reduction} = \mbox{`sum'.}
#' \end{array}
#' }
#'
#' See also [nn_triplet_margin_loss()], which computes the triplet
#' loss for input tensors using the \eqn{l_p} distance as the distance function.
#'
#' @param distance_function (callable, optional): A nonnegative, real-valued function that
#'   quantifies the closeness of two tensors. If not specified,
#'   [nn_pairwise_distance()] will be used.  Default: `None`
#' @param margin (float, optional): A non-negative margin representing the minimum difference
#'   between the positive and negative distances required for the loss to be 0. Larger
#'   margins penalize cases where the negative examples are not distant enough from the
#'   anchors, relative to the positives. Default: \eqn{1}.
#' @param swap (bool, optional): Whether to use the distance swap described in the paper
#'   [Learning shallow convolutional feature descriptors with triplet losses](http://www.bmva.org/bmvc/2016/papers/paper119/index.html) by
#'   V. Balntas, E. Riba et al. If TRUE, and if the positive example is closer to the
#'   negative example than the anchor is, swaps the positive example and the anchor in
#'   the loss computation. Default: `FALSE`.
#' @param reduction (string, optional): Specifies the (optional) reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the sum of the output will be divided by the number of
#'   elements in the output, `'sum'`: the output will be summed. Default: `'mean'`
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where \eqn{*} represents any number of additional dimensions
#'   as supported by the distance function.
#' - Output: A Tensor of shape \eqn{(N)} if `reduction` is `'none'`, or a scalar
#'   otherwise.
#'
#' @examples
#' # Initialize embeddings
#' embedding <- nn_embedding(1000, 128)
#' anchor_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
#' positive_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
#' negative_ids <- torch_randint(1, 1000, 1, dtype = torch_long())
#' anchor <- embedding(anchor_ids)
#' positive <- embedding(positive_ids)
#' negative <- embedding(negative_ids)
#'
#' # Built-in Distance Function
#' triplet_loss <- nn_triplet_margin_with_distance_loss(
#'   distance_function = nn_pairwise_distance()
#' )
#' output <- triplet_loss(anchor, positive, negative)
#'
#' # Custom Distance Function
#' l_infinity <- function(x1, x2) {
#'   torch_max(torch_abs(x1 - x2), dim = 1)[[1]]
#' }
#'
#' triplet_loss <- nn_triplet_margin_with_distance_loss(
#'   distance_function = l_infinity, margin = 1.5
#' )
#' output <- triplet_loss(anchor, positive, negative)
#'
#' # Custom Distance Function (Lambda)
#' triplet_loss <- nn_triplet_margin_with_distance_loss(
#'   distance_function = function(x, y) {
#'     1 - nnf_cosine_similarity(x, y)
#'   }
#' )
#'
#' output <- triplet_loss(anchor, positive, negative)
#' @export
nn_triplet_margin_with_distance_loss <- nn_module(
  "nn_triplet_margin_with_distance_loss",
  inherit = nn_loss,
  initialize = function(distance_function = NULL, margin = 1, swap = FALSE,
                        reduction = "mean") {
    super$initialize(reduction = reduction)
    if (is.null(distance_function)) {
      self$distance_function <- nn_pairwise_distance()
    } else {
      self$distance_function <- distance_function
    }
    self$margin <- margin
    self$swap <- swap
  },
  forward = function(anchor, positive, negative) {
    nnf_triplet_margin_with_distance_loss(
      anchor, positive, negative,
      distance_function = self$distance_function,
      margin = self$margin, swap = self$swap, reduction = self$reduction
    )
  }
)

#' The Connectionist Temporal Classification loss.
#'
#' Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
#' probability of possible alignments of input to target, producing a loss value which is differentiable
#' with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
#' limits the length of the target sequence such that it must be \eqn{\leq} the input length.
#'
#'
#' @param blank (int, optional): blank label. Default \eqn{0}.
#' @param reduction (string, optional): Specifies the reduction to apply to the output:
#'   `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied,
#'   `'mean'`: the output losses will be divided by the target lengths and
#'   then the mean over the batch is taken. Default: `'mean'`
#' @param zero_infinity (bool, optional):
#'   Whether to zero infinite losses and the associated gradients.
#'   Default: `FALSE`
#'   Infinite losses mainly occur when the inputs are too short
#'   to be aligned to the targets.
#'
#' @section Shape:
#' - Log_probs: Tensor of size \eqn{(T, N, C)},
#'   where \eqn{T = \mbox{input length}},
#'   \eqn{N = \mbox{batch size}}, and
#'   \eqn{C = \mbox{number of classes (including blank)}}.
#'   The logarithmized probabilities of the outputs (e.g. obtained with
#'   [nnf)log_softmax()]).
#' - Targets: Tensor of size \eqn{(N, S)} or
#'   \eqn{(\mbox{sum}(\mbox{target\_lengths}))},
#'   where \eqn{N = \mbox{batch size}} and
#'   \eqn{S = \mbox{max target length, if shape is } (N, S)}.
#'   It represent the target sequences. Each element in the target
#'   sequence is a class index. And the target index cannot be blank (default=0).
#'   In the \eqn{(N, S)} form, targets are padded to the
#'   length of the longest sequence, and stacked.
#'   In the \eqn{(\mbox{sum}(\mbox{target\_lengths}))} form,
#'   the targets are assumed to be un-padded and
#'   concatenated within 1 dimension.
#' - Input_lengths: Tuple or tensor of size \eqn{(N)},
#'   where \eqn{N = \mbox{batch size}}. It represent the lengths of the
#'   inputs (must each be \eqn{\leq T}). And the lengths are specified
#'   for each sequence to achieve masking under the assumption that sequences
#'   are padded to equal lengths.
#' - Target_lengths: Tuple or tensor of size \eqn{(N)},
#'   where \eqn{N = \mbox{batch size}}. It represent lengths of the targets.
#'   Lengths are specified for each sequence to achieve masking under the
#'   assumption that sequences are padded to equal lengths. If target shape is
#'   \eqn{(N,S)}, target_lengths are effectively the stop index
#'   \eqn{s_n} for each target sequence, such that `target_n = targets[n,0:s_n]` for
#'   each target in a batch. Lengths must each be \eqn{\leq S}
#'   If the targets are given as a 1d tensor that is the concatenation of individual
#'   targets, the target_lengths must add up to the total length of the tensor.
#' - Output: scalar. If `reduction` is `'none'`, then
#'   \eqn{(N)}, where \eqn{N = \mbox{batch size}}.
#'
#' @examples
#' # Target are to be padded
#' T <- 50 # Input sequence length
#' C <- 20 # Number of classes (including blank)
#' N <- 16 # Batch size
#' S <- 30 # Target sequence length of longest target in batch (padding length)
#' S_min <- 10 # Minimum target length, for demonstration purposes
#'
#' # Initialize random batch of input vectors, for *size = (T,N,C)
#' input <- torch_randn(T, N, C)$log_softmax(2)$detach()$requires_grad_()
#'
#' # Initialize random batch of targets (0 = blank, 1:C = classes)
#' target <- torch_randint(low = 1, high = C, size = c(N, S), dtype = torch_long())
#'
#' input_lengths <- torch_full(size = c(N), fill_value = TRUE, dtype = torch_long())
#' target_lengths <- torch_randint(low = S_min, high = S, size = c(N), dtype = torch_long())
#' ctc_loss <- nn_ctc_loss()
#' loss <- ctc_loss(input, target, input_lengths, target_lengths)
#' loss$backward()
#'
#'
#' # Target are to be un-padded
#' T <- 50 # Input sequence length
#' C <- 20 # Number of classes (including blank)
#' N <- 16 # Batch size
#'
#' # Initialize random batch of input vectors, for *size = (T,N,C)
#' input <- torch_randn(T, N, C)$log_softmax(2)$detach()$requires_grad_()
#' input_lengths <- torch_full(size = c(N), fill_value = TRUE, dtype = torch_long())
#'
#' # Initialize random batch of targets (0 = blank, 1:C = classes)
#' target_lengths <- torch_randint(low = 1, high = T, size = c(N), dtype = torch_long())
#' target <- torch_randint(
#'   low = 1, high = C, size = as.integer(sum(target_lengths)),
#'   dtype = torch_long()
#' )
#' ctc_loss <- nn_ctc_loss()
#' loss <- ctc_loss(input, target, input_lengths, target_lengths)
#' loss$backward()
#' @references
#' A. Graves et al.: Connectionist Temporal Classification:
#' Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
#' https://www.cs.toronto.edu/~graves/icml_2006.pdf
#'
#' @note
#' In order to use CuDNN, the following must be satisfied: `targets` must be
#' in concatenated format, all `input_lengths` must be `T`.  \eqn{blank=0},
#' `target_lengths` \eqn{\leq 256}, the integer arguments must be of
#' The regular implementation uses the (more common in PyTorch) `torch_long` dtype.
#' dtype `torch_int32`.
#'
#' @note
#' In some circumstances when using the CUDA backend with CuDNN, this operator
#' may select a nondeterministic algorithm to increase performance. If this is
#' undesirable, you can try to make the operation deterministic (potentially at
#' a performance cost) by setting `torch.backends.cudnn.deterministic = TRUE`.
#'
#' @export
nn_ctc_loss <- nn_module(
  "nn_ctc_loss",
  inherit = nn_loss,
  initialize = function(blank = 0, reduction = "mean", zero_infinity = FALSE) {
    super$initialize(reduction = reduction)
    self$blank <- blank
    self$zero_infinity <- zero_infinity
  },
  forward = function(log_probs, targets, input_lengths, target_lengths) {
    nnf_ctc_loss(
      log_probs, targets, input_lengths, target_lengths,
      blank = self$blank,
      reduction = self$reduction, zero_infinity = self$zero_infinity
    )
  }
)
