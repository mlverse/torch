#' @include optim.R
NULL

optim_Adam <- R6::R6Class(
  "optim_adam", 
  lock_objects = FALSE,
  inherit = Optimizer,
  public = list(
    initialize = function(params, lr=1e-3, betas=c(0.9, 0.999), eps=1e-8,
                          weight_decay=0, amsgrad=FALSE) {
      
      
      if (lr < 0)
        value_error("Invalid learning rate: {lr}")
      
      if (eps < 0)
        value_error("Invalid eps: {eps}")
      
      if (betas[[1]] < 0 || betas[[1]] > 1)
        value_error("Invalid beta parameter at index 1")
      
      if (betas[[2]] < 0 || betas[[2]] > 1)
        value_error("Invalid beta parameter at index 2")
      
      if (weight_decay < 0)
        value_error("Invalid weight decay value: {weight_decay}")
      
      defaults <- list(lr=lr, betas=betas, eps = eps, weight_decay = weight_decay,
                       amsgrad = amsgrad)
      
      super$initialize(params, defaults)
    },
    
    step = function(closure = NULL) {
      with_no_grad({
        
        loss <- NULL
        if (!is.null(closure)) {
          with_enable_grad({
            loss <- closure()
          })
        }
        
        for (g in seq_along(self$param_groups)) {
          
          group <- self$param_groups[[g]]
          
          for (p in seq_along(group$params)) {
            
            param <- group$params[[p]]
            
            if (is.null(param$grad))
              next
            
            grad <- param$grad
            
            # if (grad$is_sparse) {
            #   runtime_error("Adam does not support sparse gradients, please consider",
            #                 "SparseAdam instead")
            # }
            amsgrad <- group$amsgrad
            
            # state initialization
            if (length(param$state) == 0) {
              param$state <- list()
              param$state[["step"]] <- 0
              param$state[["exp_avg"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              param$state[["exp_avg_sq"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              if (amsgrad) {
                param$state[['max_exp_avg_sq']] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              }
            }
            
            exp_avg <- param$state[["exp_avg"]]
            exp_avg_sq <- param$state[["exp_avg_sq"]]
            if (amsgrad) {
              max_exp_avg_sq <- param$state[['max_exp_avg_sq']]
            }
            beta1 <- group$betas[[1]]
            beta2 <- group$betas[[2]]
            
            param$state[["step"]] <- param$state[["step"]] + 1
            bias_correction1 <- 1 - beta1 ^ param$state[['step']]
            bias_correction2 <- 1 - beta2 ^ param$state[['step']]
            
            if (group$weight_decay != 0) {
              grad$add_(p, alpha=group$weight_decay)
            }
            
            # Decay the first and second moment running average coefficient
            exp_avg$mul_(beta1)$add_(grad, alpha=1 - beta1)
            exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value=1 - beta2)
            
            if (amsgrad) {
              
              # Maintains the maximum of all 2nd moment running avg. till now
              max_exp_avg_sq$set_data(max_exp_avg_sq$max(other = exp_avg_sq))
              # Use the max. for normalizing running avg. of gradient
              denom <- (max_exp_avg_sq$sqrt() / sqrt(bias_correction2))$add_(group$eps)
            } else {
              denom <- (exp_avg_sq$sqrt() / sqrt(bias_correction2))$add_(group$eps)
            }
            
            step_size <- group$lr / bias_correction1
              
            param$addcdiv_(exp_avg, denom, value=-step_size)
          }
        }
      })
      loss
    }
  )
)

#' @export
optim_adam <- function(params, lr=1e-3, betas=c(0.9, 0.999), eps=1e-8,
                       weight_decay=0, amsgrad=FALSE) {
  optim_Adam$new(params, lr, betas, eps, weight_decay, amsgrad)
}