#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-gamma.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Chi2 <- R6::R6Class(
  "torch_Chi2",
  lock_objects = FALSE,
  inherit = Gamma,
  public = list(
    .arg_constraints = list(df = constraint_positive),
    initialize = function(df, validate_args = NULL) {
      super$initialize(0.5 * df, torch_tensor(0.5), validate_args = validate_args)
    },
    expand = function(batch_shape, .instance = NULL) {
      new <- private$.get_checked_instance(self, .instance, NULL)
      new$.__enclos_env__$super$expand(batch_shape)
    }
  ),
  active = list(
    df = function() {
      self$concentration * 2
    }
  )
)

Chi2 <- add_class_definition(Chi2)

#' Creates a Chi2 distribution parameterized by shape parameter `df`.
#' This is exactly equivalent to `distr_gamma(alpha=0.5*df, beta=0.5)`
#'
#' @param df (float or torch_tensor): shape parameter of the distribution
#' @inheritParams distr_bernoulli
#'
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#'
#' @examples
#' m <- distr_chi2(torch_tensor(1.0))
#' m$sample() # Chi2 distributed with shape df=1
#' torch_tensor(0.1046)
#' @export
distr_chi2 <- function(df, validate_args = NULL) {
  Chi2$new(df, validate_args = validate_args)
}
