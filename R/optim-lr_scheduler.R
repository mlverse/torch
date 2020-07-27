#' @include optim.R
NULL

LRSchduler <- R6::R6Class(
  "LRScheduler",
  lock_objects = FALSE,
  initialize = function(optmizer, last_epoch = -1, verbose = FALSE) {
    
    if (!is_torch_optimizer(optimizer))
      type_error("not an optimizer")
    
    self$optimizer <- optimizer
    
    if (last_epoch == -1) {
      
      optimizer$param_groups <- lapply(
        optimizer$param_groups,
        function(group) {
          if (is.null(group[["initial_lr"]]))
            group[["initial_lr"]] <- group[["lr"]]  
          
          group
        }
      )
      
    } else {
      
      lapply(
        optimizer$param_groups,
        function(group) {
          value_error("param 'inital_lr' not is not specified.")
        }
      )
      
    }
    
    self$base_lrs <- lapply(optimizer$param_groups, 
                            function(group) group[["initial_lr"]])
    
    self$last_epoch <- last_epoch
    
  }
)