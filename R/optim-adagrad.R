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
          
          params_with_grad <- list()
          grads <- list()
          state_sums  <- list()
          state_steps <- list()
          
          for (p in seq_along(group$params)) {
            param <- group$params[[p]]
            
            if (!is.null(param$grad)) {
              params_with_grad[[p]] <- p
              grads[[p]] <- p$grad
              state_sums[[p]] <- param$state[['sum']]
              param$state[['step']] <- param$state[['step']] + 1
              state_steps[[p]] <- param$state[['step']]
            }
          }
        }
        
       nnf_adagrad(
         params_with_grad,
         grads,
         state_sums,
         state_steps,
         group['lr'],
         group['weight_decay'],
         group['lr_decay'],
         group['eps']
       )
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
