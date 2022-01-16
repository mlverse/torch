#' @include distributions.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Categorical <- R6::R6Class(
  "torch_Categorical",
  lock_objects = FALSE,
  inherit = Distribution,
  public = list(
    .arg_constraints = list(
      probs = constraint_simplex,
      logits = constraint_real_vector
    ),
    has_enumerate_support = TRUE,
    initialize = function(probs = NULL, logits = NULL, validate_args = NULL) {
      if (is.null(probs) == is.null(logits)) {
        value_error("Either probs or logits must be specified but not both.")
      }

      if (!is.null(probs)) {
        if (probs$dim() < 1) {
          value_error("`probs` must be at least one-dimensional.")
        }

        self$.probs <- probs / probs$sum(-1, keepdim = TRUE)
      } else {
        if (logits$dim() < 1) {
          value_error("`logits` must be at least one-dimensional.")
        }

        self$.logits <- logits - logits$logsumexp(dim = -1, keepdim = TRUE)
      }

      self$.param <- if (!is.null(self$.probs)) self$.probs else self$.logits
      self$.num_events <- self$.param$size(-1)
      batch_shape <- if (self$.param$ndim > 1) head2(self$.param$shape, -1) else list()

      super$initialize(batch_shape, validate_args = validate_args)
    },
    expand = function(batch_shape, instance = NULL) {
      param_shape <- c(batch_shape, self$.num_events)

      .args <- list(probs = NULL, logits = NULL)

      if (!is.null(self$probs)) {
        .args$probs <- self$probs$expand(param_shape)
      }

      if (!is.null(self$logits)) {
        .args$logits <- self$logits$expand(param_shape)
      }

      .args$validate_args <- self$.validate_args

      new <- private$.get_checked_instance(self, .instance, .args)
      new
    },
    sample = function(sample_shape = list()) {
      probs2d <- self$probs$reshape(c(-1, self$.num_events))
      numel <- prod(as.integer(sample_shape))
      samples_2d <- torch_multinomial(probs2d, numel, replacement = TRUE)$t()
      samples_2d$reshape(self$.extended_shape(sample_shape))
    },
    log_prob = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }

      value <- value$to(dtype = torch_long())$unsqueeze(-1)
      out <- torch_broadcast_tensors(list(value, self$logits))
      value <- out[[1]]
      log_pmf <- out[[2]]
      value <- value[.., 1, drop = FALSE]
      log_pmf$gather(-1, value)$squeeze(-1)
    }
  ),
  active = list(
    probs = function() {
      if (!is.null(self$.probs)) {
        self$.probs
      } else {
        logits_to_probs(self$.logits)
      }
    },
    logits = function() {
      if (!is.null(self$.logits)) {
        self$.logits
      } else {
        probs_to_logits(self$.probs)
      }
    },
    mean = function() {
      torch_full(self$.extended_shape(), NaN,
        dtype = self$probs$dtype,
        device = self$probs$device
      )
    },
    variance = function() {
      torch_full(self$.extended_shape(), NaN,
        dtype = self$probs$dtype,
        device = self$probs$device
      )
    }
  )
)

Categorical <- add_class_definition(Categorical)

#' Creates a categorical distribution parameterized by either `probs` or
#' `logits` (but not both).
#'
#' @note
#' It is equivalent to the distribution that [torch_multinomial()]
#' samples from.
#'
#' Samples are integers from \eqn{\{0, \ldots, K-1\}} where `K` is `probs$size(-1)`.
#'
#' If `probs` is 1-dimensional with length-`K`, each element is the relative probability
#' of sampling the class at that index.
#'
#' If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
#' relative probability vectors.
#'
#' @note The `probs` argument must be non-negative, finite and have a non-zero sum,
#' and it will be normalized to sum to 1 along the last dimension. attr:`probs`
#' will return this normalized value.
#' The `logits` argument will be interpreted as unnormalized log probabilities
#' and can therefore be any real number. It will likewise be normalized so that
#' the resulting probabilities sum to 1 along the last dimension. attr:`logits`
#' will return this normalized value.
#'
#' See also: [torch_multinomial()]
#'
#'
#' @param probs (Tensor): event probabilities
#' @param logits (Tensor): event log probabilities (unnormalized)
#' @inheritParams distr_normal
#'
#' @examples
#' m <- distr_categorical(torch_tensor(c(0.25, 0.25, 0.25, 0.25)))
#' m$sample() # equal probability of 1,2,3,4
#' @export
distr_categorical <- function(probs = NULL, logits = NULL, validate_args = NULL) {
  Categorical$new(probs, logits, validate_args = validate_args)
}
