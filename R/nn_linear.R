nn_linear <- nn_module(
  "nn_linear",
  initialize = function(in_features, out_features, bias = TRUE) {
    self$in_features <- in_features
    self$out_feature <- out_features
    
    self$weight <- nn_parameter(torch_empty(out_features, in_features))
    if (bias)
      self$bias <- nn_parameter(torch_empty(out_features))
    else
      self$bias <- NULL
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    
  },
  forward = function() {
    
  }
)