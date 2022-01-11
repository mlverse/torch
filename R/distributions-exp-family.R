#' @include utils.R

#' ExponentialFamily is the abstract base class for probability distributions belonging to an
#' exponential family, whose probability mass/density function has the form is defined below
#'
#' \deqn{
#'   p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))
#' }
#' where \eqn{\theta} denotes the natural parameters, \eqn{t(x)} denotes the sufficient statistic,
#' \eqn{F(\theta)} is the log normalizer function for a given family and \eqn{k(x)} is the carrier
#' measure.
#'
#' @note
#' This class is an intermediary between the `Distribution` class and distributions which belong
#' to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
#' divergence methods. We use this class to compute the entropy and KL divergence using the AD
#' framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
#' Cross-entropies of Exponential Families).
ExponentialFamily <- R6::R6Class(
  "torch_ExponentialFamily",
  lock_objects = FALSE,
  inherit = Distribution,
  public = list(

    #' Method to compute the entropy using Bregman divergence of the log normalizer.
    entropy = function() {
      result <- -self$.mean_carrier_measure
      nparams <- Map(
        function(x) x$detach()$requires_grad_(),
        self$.natural_params,
      )
      lg_normal <- self$.log_normalizer(nparams)
      gradients <- autograd_grad(lg_normal$sum(), nparams, create_graph = TRUE)
      result <- result + lg_normal

      for (i in seq_along(nparams)) {
        np <- nparams[[i]]
        g <- gradients[[i]]
        result <- result - np * g
      }

      result
    }
  ),
  private = list(
    #' Abstract method for log normalizer function.
    #' Returns a log normalizer based on the distribution and input
    .log_normalizer = function(natural_params) {
      not_implemented_error()
    }
  ),
  active = list(

    #'  Abstract method for natural parameters.
    #'  Returns a tuple of Tensors based on the distribution
    .natural_params = function() {
      not_implemented_error()
    },

    #' Abstract method for expected carrier measure,
    #' which is required for computing entropy.
    .mean_carrier_measure = function() {
      not_implemented_error()
    }
  )
)

ExponentialFamily <- add_class_definition(ExponentialFamily)
