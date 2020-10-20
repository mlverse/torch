#' @include optim.R
NULL

optim_ASGD <- R6::R6Class(
  "optim_asgd", 
  lock_objects = FALSE,
  inherit = Optimizer,
  public = list(
    initialize = function(params, lr=1e-2, lambda=1e-4, 
                          alpha=0.75, t0=1e6, weight_decay=0) {
      
      
      if (lr < 0)
        value_error("Invalid learning rate: {lr}")
      
      if (weight_decay < 0)
        value_error("Invalid weight_decay value: {weight_decay}")

      defaults <- list(lr=lr, lambda=lambda, alpha=alpha, 
                       t0=t0, weight_decay=weight_decay)
      
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
          weight_decay <- group$weight_decay

          for (p in seq_along(group$params)) {
            
            param <- group$params[[p]]
            
            if (is.null(param$grad) || is_undefined_tensor(param$grad))
              next
            
            grad <- param$grad
            
            if (length(param$state) == 0) {
              param$state[["step"]] <- 0
              param$state[["eta"]] <- group[["lr"]]
              param$state[["mu"]] <- 1
              param$state[["ax"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
            }
            
            param$state[["step"]] <- param$state[["step"]] + 1
            
            if (group[["weight_decay"]] != 0)
              grad <- grad$add(param, alpha=group$weight_decay)
            
            # decay term
            param$mul_(1 - group$lambda * param$state$eta)
            
            # update parameter
            param$add_(grad, alpha=-param$state$eta)
            
            # averaging
            if (param$state[["mu"]] != 1)
              param$state[["mu"]]$add_(param$sub(param$state[["ax"]])$mul(param$state[["mu"]]))
            else
              param$state[["ax"]]$copy_(param)
            
            # update eta and mu
            denominator <- (1 + group[["lambda"]] * group[["lr"]] * param$state[["step"]]) ^ group[["alpha"]]
            param$state[["eta"]] <- group[["lr"]] / denominator
            param$state[["mu"]] <- 1 / max(1, param$state[["step"]]- group[["t0"]])
          }
        }
      })
      loss
    }
  )
)

#' Averaged Stochastic Gradient Descent optimizer
#' 
#' Proposed in [Acceleration of stochastic approximation by averaging](https://dl.acm.org/citation.cfm?id=131098)
#' 
#' @param params (iterable): iterable of parameters to optimize or lists defining
#'   parameter groups
#' @param lr (float): learning rate
#' @param lambda (float, optional): decay term (default: 1e-4)
#' @param alpha (float, optional): power for eta update (default: 0.75)
#' @param t0 (float, optional): point at which to start averaging (default: 1e6)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' 
#' @examples
#' \dontrun{
#' optimizer <- optim_asgd(model$parameters(), lr=0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
#' 
#' @export
optim_asgd <- function(params,  lr=1e-2, lambda=1e-4, 
                       alpha=0.75, t0=1e6, weight_decay=0) {
  optim_ASGD$new(params, lr, lambda, alpha,
                 t0, weight_decay)
}
