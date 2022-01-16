#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Poisson <- R6::R6Class(
  "torch_Poisson",
  lock_objects = FALSE,
  inherit = ExponentialFamily,
  public = list(
    initialize = function(rate, validate_args = NULL) {
      self$rate <- broadcast_all(list(rate))[[1]]
      if (inherits(rate, "numeric")) {
        batch_shape <- 1
      } else {
        batch_shape <- self$rate$size()
      }
      super$initialize(batch_shape, validate_args = validate_args)
    },
    expand = function(batch_shape, .instance) {
      new <- private$.get_checked_instance(self, .instance)
      new$rate <- self$rate$expand(batch_shape)
      new$.__enclos_env__$super$initialize(batch_shape, validate_args = FALSE)
      new$.validate_args <- self$.validate_args
      new
    },
    sample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)
      with_no_grad({
        torch_poisson(self$rate$expand(shape))
      })
    },
    log_prob = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      results <- broadcast_all(list(self$rate, value))
      rate <- results[[1]]
      value <- results[[2]]
      (rate$log() * value) - rate - (value + 1)$lgamma()
    }
  ),
  private = list(
    .log_normalizer = function(x) {
      torch_exp(x)
    }
  ),
  active = list(
    mean = function() {
      self$rate
    },
    variance = function() {
      self$rate
    },
    .natural_params = function() {
      torch_log(self$rate)
    }
  )
)

#' Creates a Poisson distribution parameterized by `rate`, the rate parameter.
#'
#' @description
#' Samples are nonnegative integers, with a pmf given by
#' \deqn{
#' \mbox{rate}^{k} \frac{e^{-\mbox{rate}}}{k!}
#' }
#'
#' @param rate (numeric, torch_tensor): the rate parameter
#' @inheritParams distr_bernoulli
#'
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#'
#' @examples
#' m <- distr_poisson(torch_tensor(4))
#' m$sample()
#' @export
distr_poisson <- function(rate, validate_args = NULL) {
  Poisson$new(rate, validate_args)
}
