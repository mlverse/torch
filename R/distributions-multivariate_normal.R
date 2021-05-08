.batch_mv <- function(bmat, bvec) {
  torch_matmul(bmat, bvec$unsqueeze(-1))$squeeze(-1)
}

.batch_mahalanobis <- function(bL, bx) {
  n <- bx$size(-1)
  bx_batch_shape <- bx$shape[-length(bx$shape)]
  
  # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
  # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
  
  bx_batch_dims <- length(bx_batch_shape)
  bL_batch_dims <- bL$dim() - 2
  outer_batch_dims <- bx_batch_dims - bL_batch_dims
  old_batch_dims <- outer_batch_dims + bL_batch_dims
  new_batch_dims <- outer_batch_dims + 2 * bL_batch_dims
  
  # Reshape bx with the shape (..., 1, i, j, 1, n)
  bx_new_shape <- head(bx$shape, outer_batch_dims)
  l <- list(
    sL = head(bL$shape, length(bL$shape)-2),
    sx = bx$shape[seq(outer_batch_dims, length(bx$shape) - 1)]
  )
  for (x in transpose_list(l)) {
    bx_new_shape <- c(bx_new_shape, c(trunc(x$sx / x$sL), x$sL))
  }
  bx_new_shape <- c(bx_new_shape, n)
  
  permute_dims <- c(seq(1, outer_batch_dims), 
                    seq(outer_batch_dims, new_batch_dims, by = 2),
                    seq(outer_batch_dims + 1, new_batch_dims, by = 2),
                    new_batch_dims)
    
  bx <- bx$permute(permute_dims)
  
  flat_L <- bL$reshape(c(-1, n, n))  # shape = b x n x n
  flat_x <- bx$reshape(c(-1, flat_L$size(1), n))  # shape = c x b x n
  flat_x_swap <- flat_x$permute(c(2, 3, 1))  # shape = b x n x c
  M_swap <- torch_triangular_solve(flat_x_swap, flat_L, upper=FALSE)[[1]]$pow(2)$sum(-2)  # shape = b x c
  M <- M_swap$t()
  
  # Now we revert the above reshape and permute operators.
  permuted_M <- M$reshape(head(bx$shape, length(bx$shape) - 1))  # shape = (..., 1, j, i, 1)
  permute_inv_dims <- seq_len(outer_batch_dims)
  
  for (i in seq_len(bL_batch_dims)) {
    permute_inv_dims <- c(permute_inv_dims, c(outer_batch_dims + i, old_batch_dims + i))
  }
  reshaped_M <- permuted_M$permute(permute_inv_dims)
  reshaped_M$reshape(bx_batch_shape)
}

MultivariateNormal <- R6::R6Class(
  "torch_MultivariateNormal",
  lock_objects = FALSE,
  inherit = Distribution,
  
  public = list(
    
    .arg_constraints = list(loc   = constraint_real, 
                            scale = constraint_positive),
    .support = constraint_real,
    has_rsample = TRUE,
    ._mean_carrier_measure = 0,
    
    initialize = function(loc, scale, validate_args = NULL){
      # TODO
      broadcasted <- broadcast_all(list(loc, scale))
      self$loc    <- broadcasted[[1]]
      self$scale  <- broadcasted[[2]]
      
      # TODO: check this fragment
      # It seems it's more suitbale for Python
      # if (inherits(loc, "numeric") & inherits(scale, "numeric"))
      #   batch_shape <- NULL
      # else
      #   batch_shape <- self$loc$size()
      
      batch_shape <- self$loc$size()
      super$initialize(batch_shape, validate_args=validate_args)
    }, 
    
    expand = function(batch_shape, .instance=NULL){
      
      .args <- list(
        loc = self$loc$expand(batch_shape),
        scale = self$scale$expand(batch_shape)
      )
      
      new <- private$.get_checked_instance(self, .instance, .args)
      
      # new$loc <- self$loc$expand(batch_shape)
      # new$scale <- self$scale$expand(batch_shape)
      
      new$.__enclos_env__$super$initialize(
        batch_shape, validate_args=FALSE
      )
      new$.validate_args <- self$.validate_args
      new
    },
    
    sample = function(sample_shape=NULL){
      shape <- self$.extended_shape(sample_shape)
      
      with_no_grad({
        torch_normal(
          self$loc$expand(shape), self$scale$expand(shape)
        )
      })
    },
    
    rsample = function(sample_shape=NULL){
      shape <- self$.extended_shape(sample_shape)
      eps <- .standard_normal(shape, dtype=self$loc$dtype, 
                              device=self$loc$device)
      self$loc + eps * self$scale
    },
    
    log_prob = function(value){
      if (self$.validate_args)
        self$.validate_sample(value)
      # compute the variance
      var <- self$scale ** 2
      
      if (inherits(self$scale, "numeric"))
        log_scale <- log(self$scale)
      else 
        log_scale <- self$scale$log()
      
      -((value - self$loc) ** 2) / (2 * var) - log_scale - log(sqrt(2 * pi))
    },
    
    cdf = function(value){
      if (self$.validate_args)
        self$.validate_sample(value)
      0.5 * (1 + torch_erf((value - self$loc) * self$scale$reciprocal() / sqrt(2)))
    },
    
    icdf = function(value){
      if (self$.validate_args)
        self$.validate_sample(value)
      self$loc + self$scale * torch_erfinv(2 * value - 1) * sqrt(2)
    },
    
    entropy = function(){
      0.5 + 0.5 * log(2 * pi) + torch_log(self$scale)
    }
    
  ), 
  
  private = list(
    .log_normalizer= function(x, y){
      -0.25 * x$pow(2) / y + 0.5 * torch_log(-pi / y)
    }
  ),
  
  active = list(
    
    mean = function(){
      self$loc
    },
    
    stddev = function(){
      self$scale
    },
    
    variance = function(){
      self$stddev$pow(2)
    },
    
    .natural_params = function(){
      list(self$loc / self$scale$pow(2), -0.5 * self$scale$pow(2)$reciprocal())
    },
    
    .mean_carrier_measure = function(){
      self$._mean_carrier_measure
    },
    
    support = function(){
      private$.support
    }
  )
)

MultivariateNormal <- add_class_definition(MultivariateNormal)