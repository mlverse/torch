#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

# TODO: consider different handling torch.Size use cases

Gamma <- R6::R6Class(
  "torch_Gamma",
  lock_objects = FALSE,
  inherit = ExponentialFamily,
  public = list(
    .arg_constraints = list(
      concentration = constraint_positive,
      rate = constraint_positive
    ),
    .support = constraint_positive,
    has_rsample = TRUE,
    ._mean_carrier_measure = 0,
    initialize = function(concentration, rate, validate_args = TRUE) {
      broadcasted <- broadcast_all(list(concentration, rate))
      self$concentration <- broadcasted[[1]]
      self$rate <- broadcasted[[2]]

      batch_shape <- self$concentration$size()
      super$initialize(batch_shape, validate_args = validate_args)
    },
    expand = function(batch_shape, .instance = NULL) {
      new <- private$.get_checked_instance(self, .instance)
      new$concentration <- self$concentration$expand(batch_shape)
      new$rate <- self$rate$expand(batch_shape)
      new$.__enclos_env__$super$initialize(batch_shape, validate_args = FALSE)
      new$.validate_args <- self$.validate_args
      new
    },
    rsample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)
      value <- torch__standard_gamma(self$concentration$expand(shape)) / self$rate$expand(shape)
      value$detach()$clamp_(min = torch_finfo(value$dtype)$tiny)
      value
    },
    log_prob = function(value) {
      if (is.numeric(value)) {
        value <- torch_tensor(value, dtype = self$rate$dtype, device = self$rate$device)
      }
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      (self$concentration * torch_log(self$rate) +
        (self$concentration - 1) * torch_log(value) -
        self$rate * value - torch_lgamma(self$concentration))
    },
    entropy = function() {
      (self$concentration - torch_log(self$rate) + torch_lgamma(self$concentration) +
        (1.0 - self$concentration) * torch_digamma(self$concentration))
    }
  ),
  private = list(
    .log_normalizer = function(x, y) {
      torch_lgamma(x + 1) + (x + 1) + torch_log(-y$reciprocal())
    }
  ),
  active = list(
    mean = function() {
      self$concentration / self$rate
    },
    variance = function() {
      self$concentration / self$rate$pow(2)
    },
    .natural_params = function() {
      list(self$concentration - 1, -self$rate)
    }
  )
)

#' Creates a Gamma distribution parameterized by shape `concentration` and `rate`.
#'
#' @param concentration (float or Tensor): shape parameter of the distribution
#' (often referred to as alpha)
#' @param rate (float or Tensor): rate = 1 / scale of the distribution
#' (often referred to as beta)
#' @inheritParams distr_bernoulli
#'
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#'
#' @examples
#' m <- distr_gamma(torch_tensor(1.0), torch_tensor(1.0))
#' m$sample() # Gamma distributed with concentration=1 and rate=1
#' @export
distr_gamma <- function(concentration, rate, validate_args = NULL) {
  Gamma$new(concentration, rate, validate_args)
}
