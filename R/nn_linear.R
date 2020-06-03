#' Linear module
#' 
#' Applies a linear transformation to the incoming data: `y = xA^T + b`
#' 
#' @param in_features size of each input sample
#' @param out_features size of each output sample
#' @param bias If set to `FALSE`, the layer will not learn an additive bias. 
#'   Default: `TRUE`
#' 
#' @export
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
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    if (!is.null(self$bias)) {
      fans <- nn_init_calculate_fan_in_and_fan_out(self$weight)
      bound = 1 / sqrt(fans[[1]])
      nn_init_uniform_(self$bias, -bound, bound)
    }
  },
  forward = function(input) {
    nnf_linear(input, self$weight, self$bias)
  }
)