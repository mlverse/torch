#' @include distributions.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Categorical <- R6::R6Class(
  "torch_Categorical",
  lock_objects = FALSE,
  inherit = Distribution,
  public = list(
    .arg_constraints = list(probs   = constraint_simplex, 
                            logits  = constraint_real_vector),
    has_enumerate_support = TRUE,
    initialize = function(probs = NULL, logits = NULL, validate_args = NULL) {
      
      if (is.null(probs) == is.null(logits))
        value_error("Either probs or logits must be specified but not both.")
      
      if (!is.null(probs)) {
        
        if (probs$dim() < 1)
          value_error("`probs` must be at least one-dimensional.")
        
        self$.probs <- probs / probs$sum(-1, keepdim = TRUE)
      } else {
        
        if (logits$dim() < 1)
          value_error("`logits` must be at least one-dimensional.")
        
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
      
      if (!is.null(self$probs))
        .args$probs <- self$probs$expand(param_shape)
      
      if (!is.null(self$logits))
        .args$logits <- self$logits$expand(param_shape)
      
      .args$validate_args = self$.validate_args
      
      new <- private$.get_checked_instance(self, .instance, .args)
      new
    },
    sample = function(sample_shape = list()) {
      probs2d <- self$probs$reshape(c(-1, self$.num_events))
      numel <- prod(as.integer(sample_shape))
      samples_2d <- torch_multinomial(probs2d, numel, replacement = TRUE)$t()
      # TODO Fixme when https://github.com/mlverse/torch/issues/574 is done
      torch::with_no_grad({
        samples_2d <- samples_2d$add_(1L)
      })
      samples_2d$reshape(self$.extended_shape(sample_shape))
    },
    log_prob = function(value) {
      if (self$.validate_args)
        self$.validate_sample(value)
      
      value <- value$to(dtype = torch_long())$unsqueeze(-1)
      out <- torch_broadcast_tensors(list(value, self$logits))
      value <- out[[1]]
      log_pmf <- out[[2]]
      value <- value[.., 1, drop=FALSE]
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
    }
  )
)

Categorical <- add_class_definition(Categorical)

distr_categorical <- function(probs = NULL, logits = NULL, validate_args = NULL){
  Categorical$new(probs, logits, validate_args = validate_args)
}