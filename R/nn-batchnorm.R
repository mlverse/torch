#' @include nn.R
NULL

nn_norm_base_ <- nn_module(
  "nn_norm_base",
  initialize = function(num_features, eps = 1e-5, momentum = 0.1, affine = TRUE,
                        track_running_stats = TRUE) {
    self$num_features <- num_features
    self$eps <- eps
    self$momentum <- momentum
    self$affine <- affine
    self$track_running_stats <- track_running_stats
    
    if (self$affine) {
      self$weight <- nn_parameter(torch_empty(num_features))
      self$bias = nn_parameter(torch_empty(num_features))
    } else {
      self$weight <- NULL
      self$bias <- NULL
    }
    
    if (self$track_running_stats) {
      self$running_mean <- nn_buffer(torch_zeros(num_features))
      self$running_var <- nn_buffer(torch_ones(num_features))
      self$num_batches_tracked <- nn_buffer(torch_tensor(0, dtype=torch_long))
    } else {
      self$running_mean <- NULL
      self$running_var <- NULL
      self$num_batches_tracked <- NULL
    }
    
    self$reset_parameters()
  },
  reset_running_stats = function() {
    if (self$track_running_stats) {
      self$running_mean$zero_()
      self$running_var$fill_(1)
      self$num_batches_tracked$zero_()
    }
  },
  reset_parameters = function() {
    self$reset_running_stats()
    if (self$affine) {
      nn_init_ones_(self$weight)
      nn_init_zeros_(self$bias)
    }
  },
  forward = NULL
)

nn_batch_norm_ <- nn_module(
  "nn_batch_norm_",
  inherit = nn_norm_base,
  initialize = function(num_features, eps=1e-5, momentum=0.1, affine=TRUE,
                        track_running_stats=TRUE) {
    super$initialize(num_features, eps, momentum, affine, track_running_stats)
  },
  forward = function(input) {
  
    if (!is.null(self$momentum)) {
      exponential_average_factor <- 0
    } else {
      exponential_average_factor <- self$momentum
    }
    
    if (self$training && self$track_running_stats) {
      if (!is.null(self$num_batches_tracked)) {
        self$num_batches_tracked$add_(1L, 1L)
        if (is.null(self$momentum)) {
          exponential_average_factor <- 1/self$num_batches_tracked
        } else{
          exponential_average_factor <- self$momentum
        }
      }
    }
    
    nnf_batch_norm(input, self$running_mean, self$running_var, self$weight, 
                   self$bias, self$training || self$track_running_stats,
                   exponential_average_factor, self$eps)
  }
)