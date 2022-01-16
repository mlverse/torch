#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Normal <- R6::R6Class(
  "torch_Normal",
  lock_objects = FALSE,
  inherit = ExponentialFamily,
  public = list(
    .arg_constraints = list(
      loc = constraint_real,
      scale = constraint_positive
    ),
    .support = constraint_real,
    has_rsample = TRUE,
    ._mean_carrier_measure = 0,
    initialize = function(loc, scale, validate_args = NULL) {
      # TODO
      broadcasted <- broadcast_all(list(loc, scale))
      self$loc <- broadcasted[[1]]
      self$scale <- broadcasted[[2]]

      # TODO: check this fragment
      # It seems it's more suitbale for Python
      # if (inherits(loc, "numeric") & inherits(scale, "numeric"))
      #   batch_shape <- NULL
      # else
      #   batch_shape <- self$loc$size()

      batch_shape <- self$loc$size()
      super$initialize(batch_shape, validate_args = validate_args)
    },
    expand = function(batch_shape, .instance = NULL) {
      .args <- list(
        loc = self$loc$expand(batch_shape),
        scale = self$scale$expand(batch_shape)
      )

      new <- private$.get_checked_instance(self, .instance, .args)

      # new$loc <- self$loc$expand(batch_shape)
      # new$scale <- self$scale$expand(batch_shape)

      new$.__enclos_env__$super$initialize(
        batch_shape,
        validate_args = FALSE
      )
      new$.validate_args <- self$.validate_args
      new
    },
    sample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)

      with_no_grad({
        torch_normal(
          self$loc$expand(shape), self$scale$expand(shape)
        )
      })
    },
    rsample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)
      eps <- .standard_normal(shape,
        dtype = self$loc$dtype,
        device = self$loc$device
      )
      self$loc + eps * self$scale
    },
    log_prob = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      # compute the variance
      var <- self$scale**2

      if (inherits(self$scale, "numeric")) {
        log_scale <- log(self$scale)
      } else {
        log_scale <- self$scale$log()
      }

      -((value - self$loc)**2) / (2 * var) - log_scale - log(sqrt(2 * pi))
    },
    cdf = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      0.5 * (1 + torch_erf((value - self$loc) * self$scale$reciprocal() / sqrt(2)))
    },
    icdf = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      self$loc + self$scale * torch_erfinv(2 * value - 1) * sqrt(2)
    },
    entropy = function() {
      0.5 + 0.5 * log(2 * pi) + torch_log(self$scale)
    }
  ),
  private = list(
    .log_normalizer = function(x, y) {
      -0.25 * x$pow(2) / y + 0.5 * torch_log(-pi / y)
    }
  ),
  active = list(
    mean = function() {
      self$loc
    },
    stddev = function() {
      self$scale
    },
    variance = function() {
      self$stddev$pow(2)
    },
    .natural_params = function() {
      list(self$loc / self$scale$pow(2), -0.5 * self$scale$pow(2)$reciprocal())
    },
    .mean_carrier_measure = function() {
      self$._mean_carrier_measure
    },
    support = function() {
      private$.support
    }
  )
)

Normal <- add_class_definition(Normal)

#' Creates a normal (also called Gaussian) distribution parameterized by
#' `loc` and `scale`.
#'
#' @param loc (float or Tensor): mean of the distribution (often referred to as mu)
#' @param scale (float or Tensor): standard deviation of the distribution (often referred to as sigma)
#' @param validate_args Additional arguments
#'
#' @return Object of `torch_Normal` class
#'
#' @examples
#' m <- distr_normal(loc = 0, scale = 1)
#' m$sample() # normally distributed with loc=0 and scale=1
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#' @export
distr_normal <- function(loc, scale, validate_args = NULL) {
  Normal$new(loc, scale, validate_args)
}
