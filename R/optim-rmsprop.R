#' @include optim.R
NULL

optim_RMSprop <- R6::R6Class(
  "optim_rmsprop",
  lock_objects = FALSE,
  inherit = Optimizer,
  public = list(
    initialize = function(params, lr=1e-2, alpha=0.99, eps=1e-8, 
                          weight_decay=0, momentum=0, centered=FALSE){
      
      if (lr < 0)
        value_error("Invalid learning rate: {lr}")
      if (eps < 0)
        value_error("Invalid epsilon value: {eps}")
      if (momentum < 0)
        value_error("Invalid momentum value: {momentum}")
      if (weight_decay < 0)
        value_error("Invalid weight_decay value: {weight_decay}")
      if (alpha < 0)
        value_error("Invalid alpha value: {alpha}")
      
      defaults <- list(
        lr = lr, alpha = alpha, eps = eps, weight_decay = weight_decay,
        momentum = momentum, centered = centered
      )
      
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
            #   runtime_error("RMSprop does not support sparse gradients")
            # }
            
            # state initialization
            if (length(param$state) == 0) {
              param$state <- list()
              param$state[["step"]] <- 0
              param$state[["square_avg"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              
              if (group$momentum > 0)
                param$state[["momentum_buffer"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
              
              if (group$centered > 0)
                param$state[["grad_avg"]] <- torch_zeros_like(param, memory_format=torch_preserve_format())
            
            }
            
            square_avg <- param$state[["square_avg"]]
            alpha <- group[["alpha"]]
            
            param$state[["step"]] <- param$state[["step"]] + 1
            
            
            if (group[["weight_decay"]] != 0)
              grad <- grad$add(p, alpha=group[["weight_decay"]])
            
            square_avg$mul_(alpha)$addcmul_(grad, grad, value=1 - alpha)
            
            if (group[["centered"]]) {
              grad_avg <- param$state[["grad_avg"]]
              grad_avg$mul_(alpha)$add_(grad, alpha=1 - alpha)
              avg <- square_avg$addcmul(grad_avg, grad_avg, value=-1)$sqrt_()$add_(group[["eps"]])
            } else {
              avg <-  square_avg$sqrt()$add_(group[["eps"]])
            }
            
            if (group[["momentum"]] > 0) {
              buf <- param$state[["momentum_buffer"]]
              buf$mul_(group[["momentum"]])$addcdiv_(grad, avg)
              param$add_(buf, alpha=-group[["lr"]])
            } else {
              param$addcdiv_(grad, avg, value=-group[["lr"]])
            }
          }
        }
      })
      loss
    }
  )
)

#' RMSprop optimizer
#' 
#' Proposed by G. Hinton in his
#' [course](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
#' 
#' @param params (iterable): iterable of parameters to optimize or list defining parameter groups
#' @param lr (float, optional): learning rate (default: 1e-2)
#' @param momentum (float, optional): momentum factor (default: 0)
#' @param alpha (float, optional): smoothing constant (default: 0.99)
#' @param eps (float, optional): term added to the denominator to improve
#' numerical stability (default: 1e-8)
#' @param centered (bool, optional) : if `TRUE`, compute the centered RMSProp,
#' the gradient is normalized by an estimation of its variance
#' weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' 
#' @note 
#' The centered version first appears in 
#' [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).
#' The implementation here takes the square root of the gradient average before
#' adding epsilon (note that TensorFlow interchanges these two operations). The effective
#' learning rate is thus \eqn{\alpha/(\sqrt{v} + \epsilon)} where \eqn{\alpha}
#' is the scheduled learning rate and \eqn{v} is the weighted moving average
#' of the squared gradient.
#' 
#' Update rule:
#' 
#' \deqn{
#' \theta_{t+1} = \theta_{t} - \frac{\eta }{\sqrt{{E[g^2]}_{t} + \epsilon}} * g_{t} 
#' }
#' 
#' @export
optim_rmsprop <- function(params, lr=1e-2, alpha=0.99, eps=1e-8, 
                          weight_decay=0, momentum=0, centered=FALSE){
  optim_RMSprop$new(params, lr, alpha, eps, weight_decay,
                    momentum, centered)
}
