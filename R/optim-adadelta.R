#' @include optim.R
NULL

optim_Adadelta <- R6::R6Class(
  "optim_adadelta",
  lock_objects = FALSE,
  inherit = Optimizer,
  
  public = list(
    
    initialize = function(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0){
      
      if (lr < 0)
        value_error("Invalid learning rate: {lr}")
      
      if (rho < 0 | rho > 1)
        value_error("Invalid rho value: {rho}")
      
      if (eps < 0)
        value_error("Invalid epsilon value: {eps}")
      
      if (weight_decay < 0)
        value_error("Invalid weight_decay value: {weight_decay}")
      
      defaults <- list(lr = lr, rho = rho, eps = eps,
                       weight_decay = weight_decay)
      
      super$initialize(params, defaults)
    },
    
    step = function(closure = NULL){
      
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
            
            if (is.null(param$grad) || is_undefined_tensor(param$grad))
              next
            
            grad <- param$grad
            
            # if (grad$is_sparse) {
            #   runtime_error("Adadelta does not support sparse gradients")
            # }
            
            # state initialization
            if (length(param$state) == 0) {
              param$state <- list()
              param$state[["step"]]       <- 0
              param$state[["square_avg"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              param$state[["acc_delta"]]  <- torch_zeros_like(param, memory_format=torch_preserve_format())
            }
           
            square_avg <- param$state[["square_avg"]]
            acc_delta  <- param$state[["acc_delta"]]
            
            rho <- group[["rho"]]
            eps <- group[["eps"]]
            
            param$state[["step"]] <- param$state[["step"]] + 1
            
            if (group[["weight_decay"]] != 0)
              grad <- grad$add(param, alpha=group[["weight_decay"]])
            
            square_avg$mul_(rho)$addcmul_(grad, grad, value=1 - rho)
            std   <- square_avg$add(eps)$sqrt_()
            delta <- acc_delta$add(eps)$sqrt_()$div_(std)$mul_(grad)
            param$add_(delta, alpha=-group[["lr"]])
            acc_delta$mul_(rho)$addcmul_(delta, delta, value=1 - rho)
             
          }
        }
      })
      loss
    }
  )
)


#' Adadelta optimizer
#' 
#' It has been proposed in [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701.pdf)
#' 
#' @param params (iterable): list of parameters to optimize or list defining
#'   parameter groups
#' @param lr (float, optional): learning rate (default: 1e-3)
#' @param rho (float, optional): coefficient used for computing a running average
#'   of squared gradients (default: 0.9)
#' @param eps (float, optional): term added to the denominator to improve
#'   numerical stability (default: 1e-6)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' 
#' @note 
#' 
#' According to the original paper, decaying average of the squared gradients
#' is computed as follows:
#' \deqn{
#' E[g^2]_{t} = \rho E[g^2]_{t- 1} + (1 - \rho){g_{t}}^2
#' }
#' 
#' RMS of previous squared gradients up to time t:
#' \deqn{
#' RMS[g_{t}] = \sqrt{E[g^2]_{t} + \epsilon }
#' }
#' 
#' Adadelta update rule:
#' \deqn{
#'  \begin{array}{ll}
#'  \Delta \theta_{t} = - \frac{RMS [\Delta \theta]_{t - 1} }{RMS[g]_{t}}
#'  \theta_{t+1} = \theta_{t} + \Delta \theta_{t}
#' \end{array}
#' }
#' 
#' @examples
#' \dontrun{
#' optimizer <- optim_adadelta(model$parameters, lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
#' 
#' @export
optim_adadelta <- function(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0){
  optim_Adadelta$new(params, lr, rho, eps, weight_decay)
}
