#' @include nn.R
NULL

nn_norm_base_ <- nn_module(
  "nn_norm_base_",
  initialize = function(num_features, eps = 1e-5, momentum = 0.1, affine = TRUE,
                        track_running_stats = TRUE) {
    self$num_features <- num_features
    self$eps <- eps
    self$momentum <- momentum
    self$affine <- affine
    self$track_running_stats <- track_running_stats

    if (self$affine) {
      self$weight <- nn_parameter(torch_empty(num_features))
      self$bias <- nn_parameter(torch_empty(num_features))
    } else {
      self$weight <- NULL
      self$bias <- NULL
    }

    if (self$track_running_stats) {
      self$running_mean <- nn_buffer(torch_zeros(num_features))
      self$running_var <- nn_buffer(torch_ones(num_features))
      self$num_batches_tracked <- nn_buffer(torch_tensor(0, dtype = torch_long()))
    } else {
      self$running_mean <- NULL
      self$running_var <- NULL
      self$num_batches_tracked <- NULL
    }

    self$reset_parameters()
  },
  reset_running_stats = function() {
    if (self$track_running_stats) {
      self$running_mean$zero_()
      self$running_var$fill_(1)
      self$num_batches_tracked$zero_()
    }
  },
  check_input_dim = function(input) {
    not_implemented_error("not implemented")
  },
  reset_parameters = function() {
    self$reset_running_stats()
    if (self$affine) {
      nn_init_ones_(self$weight)
      nn_init_zeros_(self$bias)
    }
  },
  .load_from_state_dict = function(state_dict, prefix) {
    num_batches_tracked_key <- paste0(prefix, "num_batches_tracked")
    if (!num_batches_tracked_key %in% names(state_dict)) {
      state_dict[[num_batches_tracked_key]] <- torch_tensor(0, dtype = torch_long())
    }

    super$.load_from_state_dict(state_dict, prefix)
  }
)

nn_batch_norm_ <- nn_module(
  "nn_batch_norm_",
  inherit = nn_norm_base_,
  initialize = function(num_features, eps = 1e-5, momentum = 0.1, affine = TRUE,
                        track_running_stats = TRUE) {
    super$initialize(num_features, eps, momentum, affine, track_running_stats)
  },
  forward = function(input) {
    self$check_input_dim(input)

    if (is.null(self$momentum)) {
      exponential_average_factor <- 0
    } else {
      exponential_average_factor <- self$momentum
    }

    if (self$training && self$track_running_stats) {
      if (!is.null(self$num_batches_tracked)) {
        self$num_batches_tracked$add_(1L, 1L)
        if (is.null(self$momentum)) {
          exponential_average_factor <- 1 / self$num_batches_tracked
        } else {
          exponential_average_factor <- self$momentum
        }
      }
    }

    if (self$training) {
      bn_training <- TRUE
    } else {
      bn_training <- is.null(self$running_mean) && is.null(self$running_var)
    }

    if (!self$training || self$track_running_stats) {
      running_mean <- self$running_mean
      running_var <- self$running_var
    } else {
      running_mean <- NULL
      running_var <- NULL
    }

    nnf_batch_norm(
      input,
      running_mean,
      running_var,
      self$weight, self$bias, bn_training, exponential_average_factor, self$eps
    )
  }
)

#' BatchNorm1D module
#'
#' Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
#' inputs with optional additional channel dimension) as described in the paper
#' [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
#'
#' \deqn{
#' y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
#' }
#'
#' The mean and standard-deviation are calculated per-dimension over
#' the mini-batches and \eqn{\gamma} and \eqn{\beta} are learnable parameter vectors
#' of size `C` (where `C` is the input size). By default, the elements of \eqn{\gamma}
#' are set to 1 and the elements of \eqn{\beta} are set to 0.
#'
#' Also by default, during training this layer keeps running estimates of its
#' computed mean and variance, which are then used for normalization during
#' evaluation. The running estimates are kept with a default :attr:`momentum`
#' of 0.1.
#' If `track_running_stats` is set to `FALSE`, this layer then does not
#' keep running estimates, and batch statistics are instead used during
#' evaluation time as well.
#'
#' @section Note:
#'
#' This `momentum` argument is different from one used in optimizer
#' classes and the conventional notion of momentum. Mathematically, the
#' update rule for running statistics here is
#' \eqn{\hat{x}_{\mbox{new}} = (1 - \mbox{momentum}) \times \hat{x} + \mbox{momentum} \times x_t},
#' where \eqn{\hat{x}} is the estimated statistic and \eqn{x_t} is the
#' new observed value.
#'
#' Because the Batch Normalization is done over the `C` dimension, computing statistics
#' on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.
#'
#' @param num_features \eqn{C} from an expected input of size
#'   \eqn{(N, C, L)} or \eqn{L} from input of size \eqn{(N, L)}
#' @param eps a value added to the denominator for numerical stability.
#'   Default: 1e-5
#' @param momentum the value used for the running_mean and running_var
#'   computation. Can be set to `NULL` for cumulative moving average
#'   (i.e. simple average). Default: 0.1
#' @param affine a boolean value that when set to `TRUE`, this module has
#'   learnable affine parameters. Default: `TRUE`
#' @param track_running_stats a boolean value that when set to `TRUE`, this
#'   module tracks the running mean and variance, and when set to `FALSE`,
#'   this module does not track such statistics and always uses batch
#'   statistics in both training and eval modes. Default: `TRUE`
#'
#' @section Shape:
#' - Input: \eqn{(N, C)} or \eqn{(N, C, L)}
#' - Output: \eqn{(N, C)} or \eqn{(N, C, L)} (same shape as input)
#'
#' @examples
#' # With Learnable Parameters
#' m <- nn_batch_norm1d(100)
#' # Without Learnable Parameters
#' m <- nn_batch_norm1d(100, affine = FALSE)
#' input <- torch_randn(20, 100)
#' output <- m(input)
#' @export
nn_batch_norm1d <- nn_module(
  "nn_batch_norm1d",
  inherit = nn_batch_norm_,
  check_input_dim = function(input) {
    if (input$dim() != 2 && input$dim() != 3) {
      value_error("expected 2D or 3D input (got {input$dim()}D input)")
    }
  }
)

#' BatchNorm2D
#'
#' Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
#' additional channel dimension) as described in the paper
#' [Batch Normalization: Accelerating Deep Network Training by Reducing
#' Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
#'
#' \deqn{
#'   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
#' }
#'
#' The mean and standard-deviation are calculated per-dimension over
#' the mini-batches and \eqn{\gamma} and \eqn{\beta} are learnable parameter vectors
#' of size `C` (where `C` is the input size). By default, the elements of \eqn{\gamma} are set
#' to 1 and the elements of \eqn{\beta} are set to 0. The standard-deviation is calculated
#' via the biased estimator, equivalent to `torch_var(input, unbiased=FALSE)`.
#' Also by default, during training this layer keeps running estimates of its
#' computed mean and variance, which are then used for normalization during
#' evaluation. The running estimates are kept with a default `momentum`
#' of 0.1.
#'
#' If `track_running_stats` is set to `FALSE`, this layer then does not
#' keep running estimates, and batch statistics are instead used during
#' evaluation time as well.
#'
#' @note
#' This `momentum` argument is different from one used in optimizer
#' classes and the conventional notion of momentum. Mathematically, the
#' update rule for running statistics here is
#' \eqn{\hat{x}_{\mbox{new}} = (1 - \mbox{momentum}) \times \hat{x} + \mbox{momentum} \times x_t},
#' where \eqn{\hat{x}} is the estimated statistic and \eqn{x_t} is the
#' new observed value.
#' Because the Batch Normalization is done over the `C` dimension, computing statistics
#' on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.
#'
#' @param num_features \eqn{C} from an expected input of size
#'  \eqn{(N, C, H, W)}
#' @param eps a value added to the denominator for numerical stability.
#'  Default: 1e-5
#' @param momentum the value used for the running_mean and running_var
#'  computation. Can be set to `None` for cumulative moving average
#'  (i.e. simple average). Default: 0.1
#' @param affine a boolean value that when set to `TRUE`, this module has
#'  learnable affine parameters. Default: `TRUE`
#' @param track_running_stats a boolean value that when set to `TRUE`, this
#'  module tracks the running mean and variance, and when set to `FALSE`,
#'  this module does not track such statistics and uses batch statistics instead
#'  in both training and eval modes if the running mean and variance are `None`.
#'  Default: `TRUE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, H, W)}
#' - Output: \eqn{(N, C, H, W)} (same shape as input)
#'
#' @examples
#' # With Learnable Parameters
#' m <- nn_batch_norm2d(100)
#' # Without Learnable Parameters
#' m <- nn_batch_norm2d(100, affine = FALSE)
#' input <- torch_randn(20, 100, 35, 45)
#' output <- m(input)
#' @export
nn_batch_norm2d <- nn_module(
  "nn_batch_norm2d",
  inherit = nn_batch_norm_,
  check_input_dim = function(input) {
    if (input$dim() != 4) {
      value_error("expected 4D input (got {input$dim()}D input)")
    }
  }
)

#' BatchNorm3D
#'
#' @description
#' Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
#'   with additional channel dimension) as described in the paper
#'   [Batch Normalization: Accelerating Deep Network Training by Reducing
#'   Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
#'
#' @details
#' \deqn{
#'   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
#' }
#'
#' The mean and standard-deviation are calculated per-dimension over the
#'   mini-batches and \eqn{\gamma} and \eqn{\beta} are learnable parameter
#'   vectors of size `C` (where `C` is the input size). By default, the elements
#'   of \eqn{\gamma} are set to 1 and the elements of \eqn{\beta} are set to
#'   0. The standard-deviation is calculated via the biased estimator,
#'   equivalent to `torch_var(input, unbiased = FALSE)`.
#'
#' Also by default, during training this layer keeps running estimates of its
#'   computed mean and variance, which are then used for normalization during
#'   evaluation. The running estimates are kept with a default `momentum`
#'   of 0.1.
#'
#' If `track_running_stats` is set to `FALSE`, this layer then does not
#'   keep running estimates, and batch statistics are instead used during
#'   evaluation time as well.
#'
#' @note
#' This `momentum` argument is different from one used in optimizer
#'   classes and the conventional notion of momentum. Mathematically, the
#'   update rule for running statistics here is:
#'   \eqn{\hat{x}_{\mbox{new}} = (1 - \mbox{momentum}) \times \hat{x} + \mbox{momentum} \times x_t},
#'   where \eqn{\hat{x}} is the estimated statistic and \eqn{x_t} is the
#'   new observed value.
#'
#' Because the Batch Normalization is done over the `C` dimension, computing
#'   statistics on `(N, D, H, W)` slices, it's common terminology to call this
#'   Volumetric Batch Normalization or Spatio-temporal Batch Normalization.
#'
#' @param num_features \eqn{C} from an expected input of size
#'  \eqn{(N, C, D, H, W)}
#' @param eps a value added to the denominator for numerical stability.
#'  Default: 1e-5
#' @param momentum the value used for the running_mean and running_var
#'  computation. Can be set to `None` for cumulative moving average
#'  (i.e. simple average). Default: 0.1
#' @param affine a boolean value that when set to `TRUE`, this module has
#'  learnable affine parameters. Default: `TRUE`
#' @param track_running_stats a boolean value that when set to `TRUE`, this
#'  module tracks the running mean and variance, and when set to `FALSE`,
#'  this module does not track such statistics and uses batch statistics instead
#'  in both training and eval modes if the running mean and variance are `None`.
#'  Default: `TRUE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, C, D, H, W)}
#' - Output: \eqn{(N, C, D, H, W)} (same shape as input)
#'
#' @examples
#' # With Learnable Parameters
#' m <- nn_batch_norm3d(100)
#' # Without Learnable Parameters
#' m <- nn_batch_norm3d(100, affine = FALSE)
#' input <- torch_randn(20, 100, 35, 45, 55)
#' output <- m(input)
#' @export
nn_batch_norm3d <- nn_module(
  "nn_batch_norm3d",
  inherit = nn_batch_norm_,
  check_input_dim = function(input) {
    if (input$dim() != 5) {
      value_error("expected 5D input (got {input$dim()}D input)")
    }
  }
)
