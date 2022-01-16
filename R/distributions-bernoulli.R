#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Bernoulli <- R6::R6Class(
  "torch_Bernoulli",
  lock_objects = FALSE,
  inherit = ExponentialFamily,
  public = list(
    .arg_constraints = list(
      probs = constraint_unit_interval,
      logits = constraint_real
    ),
    .support = constraint_real,
    has_rsample = TRUE,
    ._mean_carrier_measure = 0,
    initialize = function(probs = NULL, logits = NULL, validate_args = NULL) {
      if (is.null(probs) == is.null(logits)) {
        value_error("Either `probs` or `logits` must be specified, but not both.")
      }

      if (!is.null(probs)) {
        is_scalar <- inherits(probs, "numeric")
        self$.probs <- broadcast_all(list(probs))[[1]]
      } else {
        is_scalar <- inherits(logits, "numeric")
        self$.logits <- broadcast_all(list(logits))[[1]]
      }

      self$.param <- if (!is.null(probs)) self$.probs else self$.logits

      if (is_scalar) {
        batch_shape <- 1
      } else {
        batch_shape <- self$.param$size()
      }

      super$initialize(batch_shape, validate_args = validate_args)
    },
    expand = function(batch_shape, .instance = NULL) {
      # TODO: consider .get_checked_instance method refactoring (remove .args)
      if (".probs" %in% names(self)) {
        .args <- list(
          probs = self$probs$expand(batch_shape)
        )
        new <- private$.get_checked_instance(self, .instance, .args)
        new$.probs <- self$probs$expand(batch_shape)
        new$.param <- new$probs
      } else {
        .args <- list(
          logits = self$logits$expand(batch_shape)
        )
        new <- private$.get_checked_instance(self, .instance, .args)
        new$.logits <- self$logits$expand(batch_shape)
        new$.param <- new$logits
        new
      }
      new$.__enclos_env__$super$initialize(batch_shape, validate_args = FALSE)
      new$.validate_args <- self$.validate_args
      new
    },
    sample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)
      with_no_grad({
        torch_bernoulli(self$probs$expand(shape))
      })
    },
    log_prob = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }
      results <- broadcast_all(list(self$logits, value))
      logits <- results[[1]]
      value <- results[[2]]
      -nnf_binary_cross_entropy_with_logits(
        logits, value,
        reduction = "none"
      )
    },
    entropy = function() {
      nnf_binary_cross_entropy_with_logits(
        self$logits, self$probs,
        reduction = "none"
      )
    },
    enumerate_support = function(expand = TRUE) {
      values <- torch_arange(
        0, 1,
        dtype = self$.param$dtype,
        device = self$.param$device
      )
      values <- values$view(c(-1, rep(1, length(self$batch_shape))))
      if (expand) {
        values <- values$expand(c(-1, self$batch_shape))
      }
      values
    }
  ),
  active = list(
    mean = function() {
      self$probs
    },
    variance = function() {
      self$probs * (1 - self$probs)
    },
    # TODO: consider an equivalent to PyTorch lazy_property
    logits = function() {
      if (is.null(self$.logits)) {
        self$.logits <- probs_to_logits(self$.probs, is_binary = TRUE)
      }
      self$.logits
    },
    probs = function() {
      if (is.null(self$.probs)) {
        self$.probs <- logits_to_probs(self$.logits, is_binary = TRUE)
      }
      self$.probs
    },
    param_shape = function() {
      self$.param$size()
    },
    .natural_params = function() {
      torch_log(self$probs / (1 - self$probs))
    }
  ),
  private = list(
    .probs = NULL,
    .logits = NULL,
    .new = function(...) {
      self$.param$new(...)
    },
    .log_normalizer = function(x) {
      torch_log(1 + torch_exp(x))
    }
  )
)

Bernoulli <- add_class_definition(Bernoulli)

#' Creates a Bernoulli distribution parameterized by `probs`
#' or `logits` (but not both).
#' Samples are binary (0 or 1). They take the value `1` with probability `p`
#' and `0` with probability `1 - p`.
#'
#' @param probs (numeric or torch_tensor): the probability of sampling `1`
#' @param logits (numeric or torch_tensor): the log-odds of sampling `1`
#' @param validate_args whether to validate arguments or not.
#'
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#'
#' @examples
#' m <- distr_bernoulli(0.3)
#' m$sample() # 30% chance 1; 70% chance 0
#' @export
distr_bernoulli <- function(probs = NULL, logits = NULL, validate_args = NULL) {
  Bernoulli$new(probs, logits, validate_args)
}
