#' @include optim.R
NULL

optim_Adagrad <- R6::R6Class(
  "optim_adagrad",
  lock_objects = FALSE,
  inherit = Optimizer,
  public = list(
    initialize = function(params, lr=1e-2, lr_decay=0, weight_decay=0, 
                          initial_accumulator_value=0, eps=1e-10){
      
      if (lr < 0)
        value_error("Invalid learning rate: {lr)}")
      
      if (lr_decay < 0) 
        value_error("Invalid lr_decay value: {lr_decay}")
      
      if (weight_decay < 0)
        value_error("Invalid weight_decay value: {weight_decay}")
      
      if (initial_accumulator_value < 0)
        value_error("Invalid initial_accumulator_value value: {initial_accumulator_value}")
      
      if (eps < 0)
        value_error("Invalid epsilon value: {eps}")
      
      defaults <- list(lr = lr, lr_decay = lr_decay, weight_decay = weight_decay,
                       initial_accumulator_value = initial_accumulator_value,
                       eps = eps)
      super$initialize(params, defaults)
      
      for (group in self$param_groups){
        for (p in seq_along(group$params)) {
          param <- group$params[[p]]
          param$state[['step']] <- 0
          param$state[['sum']]  <- torch_full_like(
            param, 
            initial_accumulator_value, 
            memory_format=torch.preserve_format()
          )
        }
      }
    },
    
    share_memory = function(){
      for (group in self$param_groups){
        for (p in seq_along(group$params)) {
          param <- group$params[[p]]
          param$state[['sum']]$share_memory_()
        }
      }
    },
    
    step = function(closure = NULL){
      with_no_grad({
        
        loss <- NULL
        if (!is.null(closure)) {
          with_enable_grad({
            loss <- closure()
          })
        }
        
        for (group in self$param_groups){
          
          for (p in seq_along(group$params)) {
            param <- group$params[[p]]
            
            if (is.null(param$grad))
              next
            
            param$state[['step']] <- param$state[['step']] + 1
            
            grad       <- param$grad
            state_sum  <- param$state[['sum']]
            state_step <- param$state[['step']]
            
           if (weight_decay != 0) {
              # if (grad$is_sparse) {
              #   runtime_error("weight_decay option is not compatible with sparse gradients")
              # }
              grad <- grad$add(param, alpha = weight_decay)
            }
            
            clr <-  lr / (1 + (step - 1) * lr_decay)
            
            if (grad$is_sparse) {
              grad <- grad$coalesce()
              grad_indices <- grad$`_indices`()
              grad_values <-  grad$`_values`()
              size <- grad$size()
              
              state_sum$add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
              std = state_sum.sparse_mask(grad)
              std_values = std._values().sqrt_().add_(eps)
              param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
            } else {
              state_sum.addcmul_(grad, grad, value=1)
              std = state_sum.sqrt().add_(eps)
              param.addcdiv_(grad, std, value=-clr)
            }
            
          }
        }
      })
      loss
    }
  )
)

#' Adagrad optimizer
#' 
#' Proposed in [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/v12/duchi11a.html)
#' 
#' @param params (iterable): list of parameters to optimize or list parameter groups
#' @param lr (float, optional): learning rate (default: 1e-2)
#' @param lr_decay (float, optional): learning rate decay (default: 0)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' @param eps (float, optional): term added to the denominator to improve
#' numerical stability (default: 1e-10)
#' 
#' @note 
#' 
#' Update rule: 
#' 
#' deqn{
#'\theta_{t+1} = \theta_{t} - \frac{\eta }{\sqrt{G_{t} + \epsilon}} \odot g_{t} 
#' }
#' 
#' 
#' 
optim_adagrad <- function(params, lr=1e-2, lr_decay=0, weight_decay=0, 
                          initial_accumulator_value=0, eps=1e-10){
  optim_Adagrad$new(params, lr, lr_decay, weight_decay, 
                    initial_accumulator_value, eps)
}
