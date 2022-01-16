#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-gamma.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

MixtureSameFamily <- R6::R6Class(
  "torch_MixtureSameFamily",
  inherit = Distribution,
  lock_objects = FALSE,
  public = list(
    initialize = function(mixture_distribution,
                          component_distribution,
                          validate_args = NULL) {
      self$.mixture_distribution <- mixture_distribution
      self$.component_distribution <- component_distribution

      if (!inherits(self$.mixture_distribution, "torch_Categorical")) {
        value_error("Mixture distribution must be distr_categorical.")
      }

      if (!inherits(self$.component_distribution, "torch_Distribution")) {
        value_error("Component distribution must be an instance of torch_Distribution.")
      }

      cdbs <- head2(self$.component_distribution$batch_shape, -1)
      km <- tail(self$.mixture_distribution$logits$shape, 1)
      kc <- tail(self$.component_distribution$batch_shape, 1)

      if (!is.null(km) && !is.null(kc) && km != kc) {
        value_error("Mixture distribution component ({km}) does not equal component_distribution$batch_shape[-1] ({kc}).")
      }

      self$.num_components <- km
      event_shape <- self$.component_distribution$event_shape
      self$.event_ndims <- length(event_shape)

      super$initialize(
        batch_shape = cdbs, event_shape = event_shape,
        validate_args = validate_args
      )
    },
    expand = function(batch_shape, instance = NULL) {
      batch_shape_comp <- c(batch_shape, self$.num_component)

      .args <- list()
      .args$mixture_distribution <- self$.mixture_distribution$expand(batch_shape)
      .args$component_distribution <- self$.component_distribution$expand(batch_shape_comp)

      new <- private$.get_checked_instance(self, instance, .args)
      new
    },
    cdf = function(x) {
      x <- self$.pad(x)
      cdf_x <- self$component_distribution$cdf(x)
      mix_prob <- self$mixture_distribution$probs
      torch_sum(cdf_x * mix_prob, dim = -1)
    },
    log_prob = function(x) {
      if (self$.validate_args) {
        self$.validate_sample(x)
      }
      x <- self$.pad(x)
      log_prob_x <- self$component_distribution$log_prob(x) # [S, B, k]
      log_mix_prob <- torch_log_softmax(self$mixture_distribution$logits,
        dim = -1
      ) # [B, k]
      torch_logsumexp(log_prob_x + log_mix_prob, dim = -1) # [S, B]
    },
    sample = function(sample_shape = list()) {
      with_no_grad({
        sample_len <- length(sample_shape)
        batch_len <- length(self$batch_shape)
        gather_dim <- sample_len + batch_len + 1L
        es <- self$event_shape

        # mixture samples [n, B]
        mix_sample <- self$mixture_distribution$sample(sample_shape)
        mix_shape <- mix_sample$shape

        # component samples [n, B, k, E]
        comp_samples <- self$component_distribution$sample(sample_shape)

        # Gather along the k dimension
        mix_sample_r <- mix_sample$reshape(
          c(mix_shape, rep(1, length(es) + 1))
        )
        mix_sample_r <- mix_sample_r$`repeat`(
          c(rep(1, length(mix_shape)), 1, es)
        )

        samples <- torch_gather(comp_samples, gather_dim, mix_sample_r)
        out <- samples$squeeze(gather_dim)
      })
      out
    },
    .pad = function(x) {
      x$unsqueeze(-1 - self$.event_ndims)
    }
  ),
  active = list(
    mixture_distribution = function() {
      self$.mixture_distribution
    },
    component_distribution = function() {
      self$.component_distribution
    }
  )
)

MixtureSameFamily <- add_class_definition(MixtureSameFamily)


#' Mixture of components in the same family
#'
#' The `MixtureSameFamily` distribution implements a (batch of) mixture
#' distribution where all component are from different parameterizations of
#' the same distribution type. It is parameterized by a `Categorical`
#' selecting distribution" (over `k` component) and a component
#' distribution, i.e., a `Distribution` with a rightmost batch shape
#' (equal to `[k]`) which indexes each (batch of) component.
#'
#' @examples
#' # Construct Gaussian Mixture Model in 1D consisting of 5 equally
#' # weighted normal distributions
#' mix <- distr_categorical(torch_ones(5))
#' comp <- distr_normal(torch_randn(5), torch_rand(5))
#' gmm <- distr_mixture_same_family(mix, comp)
#' @param mixture_distribution `torch_distributions.Categorical`-like
#'   instance. Manages the probability of selecting component.
#'   The number of categories must match the rightmost batch
#'   dimension of the `component_distribution`. Must have either
#'   scalar `batch_shape` or `batch_shape` matching
#'   `component_distribution.batch_shape[:-1]`
#' @param component_distribution `torch_distributions.Distribution`-like
#'   instance. Right-most batch dimension indexes component.
#' @inheritParams distr_normal
#'
#' @export
distr_mixture_same_family <- function(mixture_distribution, component_distribution,
                                      validate_args = NULL) {
  MixtureSameFamily$new(
    mixture_distribution = mixture_distribution,
    component_distribution = component_distribution,
    validate_args = validate_args
  )
}
