#' Dummy value indicating a required value.
#' 
#' export
optim_required <- function() {
  structure(list(), class = "optim_required")
}

is_optim_required <- function(x) {
  inherits(x, "optim_required")
}


Optimizer <- R6::R6Class(
  "torch_Optimizer",
  lock_objects = FALSE,
  public = list(
    initialize = function(params, defaults) {
      self$defaults <- defaults
      self$state <- list()
      self$param_groups = list()
      
      if (is_torch_tensor(params))
        param_groups <- list(list(params = list(params)))
      else if (is.list(params) && is_torch_tensor(params[[1]]))
        param_groups <- list(list(params = params))
      else if (rlang::is_named(params[[1]]))
        param_groups <- params
      else
        value_error("Wrong parameters specification.")
      
      for (p in param_groups) {
        self$add_param_group(p)
      }
    
    },
    add_param_group = function(param_group) {
      
      if (!rlang::is_named(param_group))
        value_error("param group is not named")
      
      params <- param_group$params
      if (is_torch_tensor(params))
        param_group$params <- list(params)
      
      for (param in param_group$params) {
        
        if (!is_torch_tensor(param))
          value_error("optimizer can only optimize Tensors, ",
                      "but one of the params is {class(param)}")
        
        if (!param$is_leaf())
          value_error("can't optimize a non-leaf Tensor")
        
      }
      
      
      for (nm in names(self$defaults)) {
        
        if (is_optim_required(self$defaults[[nm]]) && !nm %in% names(param_group)) {
          value_error("parameter group didn't specify a value of required ",
                      "optimization parameter {name}")
        } else if (!nm %in% names(param_group)) {
          param_group[[nm]] <- self$defaults[[nm]]
        }
          
      }
      
      # TODO: check for duplicated parameters
      
      self$param_groups <- append(self$param_groups, list(param_group))
    },
    zero_grad = function() {
      for (group in self$param_groups) {
        for (p in group$params) {
          if (!is_undefined_tensor(p$grad)) {
            p$grad$detach_()
            p$grad$zero_()
          }
        }
      }
    }
  )
)