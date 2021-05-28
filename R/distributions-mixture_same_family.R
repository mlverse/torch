#' @include distributions.R
#' @include distributions-exp-family.R
#' @include distributions-gamma.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

MixtureSameFamily <- R6::R6Class(
  "torch_MixtureSameFamily",
  public = list(
    initialize = function(self,
                          mixture_distribution,
                          component_distribution,
                          validate_args=NULL) {
      
      self$.mixture_distribution <- mixture_distribution
      self$.component_distribution <- component_distribution
      
      if (!inherits(self$.mixture_distribution, "torch_Categorical"))
        value_error("Mixture distribution must be distr_categorical.")
      
    }  
  )
)