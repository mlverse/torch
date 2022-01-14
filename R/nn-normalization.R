#' Layer normalization
#'
#' Applies Layer Normalization over a mini-batch of inputs as described in
#' the paper [Layer Normalization](https://arxiv.org/abs/1607.06450)
#'
#' \deqn{
#'   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
#' }
#'
#' The mean and standard-deviation are calculated separately over the last
#' certain number dimensions which have to be of the shape specified by
#' `normalized_shape`.
#'
#' \eqn{\gamma} and \eqn{\beta} are learnable affine transform parameters of
#' `normalized_shape` if `elementwise_affine` is `TRUE`.
#'
#' The standard-deviation is calculated via the biased estimator, equivalent to
#' `torch_var(input, unbiased=FALSE)`.
#'
#' @note Unlike Batch Normalization and Instance Normalization, which applies
#'   scalar scale and bias for each entire channel/plane with the
#'   `affine` option, Layer Normalization applies per-element scale and
#'   bias with `elementwise_affine`.
#'
#' This layer uses statistics computed from input data in both training and
#' evaluation modes.
#'
#' @param normalized_shape (int or list): input shape from an expected input
#'   of size
#'   \eqn{[* \times \mbox{normalized\_shape}[0] \times \mbox{normalized\_shape}[1] \times \ldots \times \mbox{normalized\_shape}[-1]]}
#'   If a single integer is used, it is treated as a singleton list, and this module will
#'   normalize over the last dimension which is expected to be of that specific size.
#' @param eps a value added to the denominator for numerical stability. Default: 1e-5
#' @param elementwise_affine a boolean value that when set to `TRUE`, this module
#'   has learnable per-element affine parameters initialized to ones (for weights)
#'   and zeros (for biases). Default: `TRUE`.
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)}
#' - Output: \eqn{(N, *)} (same shape as input)
#'
#' @examples
#'
#' input <- torch_randn(20, 5, 10, 10)
#' # With Learnable Parameters
#' m <- nn_layer_norm(input$size()[-1])
#' # Without Learnable Parameters
#' m <- nn_layer_norm(input$size()[-1], elementwise_affine = FALSE)
#' # Normalize over last two dimensions
#' m <- nn_layer_norm(c(10, 10))
#' # Normalize over last dimension of size 10
#' m <- nn_layer_norm(10)
#' # Activating the module
#' output <- m(input)
#' @export
nn_layer_norm <- nn_module(
  "nn_layer_norm",
  initialize = function(normalized_shape, eps = 1e-5, elementwise_affine = TRUE) {
    self$normalized_shape <- as.list(normalized_shape)
    self$eps <- eps
    self$elementwise_affine <- elementwise_affine

    if (self$elementwise_affine) {
      self$weight <- nn_parameter(torch_empty(size = self$normalized_shape))
      self$bias <- nn_parameter(torch_empty(size = self$normalized_shape))
    } else {
      self$weight <- NULL
      self$bias <- NULL
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    if (self$elementwise_affine) {
      nn_init_ones_(self$weight)
      nn_init_zeros_(self$bias)
    }
  },
  forward = function(input) {
    nnf_layer_norm(input, self$normalized_shape, self$weight, self$bias, self$eps)
  }
)

#' Group normalization
#'
#' Applies Group Normalization over a mini-batch of inputs as described in
#' the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
#'
#' \deqn{
#'   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
#' }
#'
#' The input channels are separated into `num_groups` groups, each containing
#' ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
#' separately over the each group. \eqn{\gamma} and \eqn{\beta} are learnable
#' per-channel affine transform parameter vectors of size `num_channels` if
#' `affine` is ``TRUE``.
#' The standard-deviation is calculated via the biased estimator, equivalent to
#' `torch_var(input, unbiased=FALSE)`.
#'
#' @note This layer uses statistics computed from input data in both training and
#' evaluation modes.
#'
#' @param num_groups (int): number of groups to separate the channels into
#' @param num_channels (int): number of channels expected in input
#' @param eps a value added to the denominator for numerical stability. Default: 1e-5
#' @param affine a boolean value that when set to ``TRUE``, this module
#'    has learnable per-channel affine parameters initialized to ones (for weights)
#'    and zeros (for biases). Default: ``TRUE``.
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, *)} where \eqn{C=\mbox{num\_channels}}
#' - Output: \eqn{(N, C, *)}` (same shape as input)
#'
#' @examples
#'
#' input <- torch_randn(20, 6, 10, 10)
#' # Separate 6 channels into 3 groups
#' m <- nn_group_norm(3, 6)
#' # Separate 6 channels into 6 groups (equivalent with [nn_instance_morm])
#' m <- nn_group_norm(6, 6)
#' # Put all 6 channels into a single group (equivalent with [nn_layer_norm])
#' m <- nn_group_norm(1, 6)
#' # Activating the module
#' output <- m(input)
#' @export
nn_group_norm <- nn_module(
  "nn_group_norm",
  initialize = function(num_groups, num_channels, eps = 1e-5, affine = TRUE) {
    self$num_groups <- num_groups
    self$num_channels <- num_channels
    self$eps <- eps
    self$affine <- affine
    if (self$affine) {
      self$weight <- nn_parameter(torch_empty(num_channels))
      self$bias <- nn_parameter(torch_empty(num_channels))
    } else {
      self$weight <- NULL
      self$bias <- NULL
    }
    self$reset_parameters()
  },
  reset_parameters = function() {
    if (self$affine) {
      nn_init_ones_(self$weight)
      nn_init_zeros_(self$bias)
    }
  },
  forward = function(input) {
    nnf_group_norm(input, self$num_groups, self$weight, self$bias, self$eps)
  }
)
