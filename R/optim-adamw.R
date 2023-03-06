#' Implements AdamW algorithm
#' 
#' For further details regarding the algorithm we refer to 
#' [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
#'
#' @inheritParams optim_adam
#'
#' @export
optim_adamw <- optimizer(
  "optim_adawm",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
                        weight_decay = 1e-2, amsgrad = FALSE) {
    
    if (lr < 0) {
      cli::cli_abort("Invalid learning rate: {lr}")
    }
    
    if (eps < 0) {
      cli::cli_abort("Invalid epsilon value: {eps}")
    }
    
    if (betas[1] > 1 || betas[1] < 0) {
      cli::cli_abort("Invalid betas[1] parameter value: {beta[1]}")
    }
    
    if (betas[2] > 1 || betas[2] < 0) {
      cli::cli_abort("Invalid betas[2] parameter value: {beta[2]}")
    }
    
    if (weight_decay < 0) {
      cli::cli_abort("Invalid weight_decay value: {weight_decay}")
    }
    
    defaults <- list(
      lr = lr, betas = betas, eps = eps, weight_decay = weight_decay,
      amsgrad = amsgrad
    )
    
    super$initialize(params, defaults)
  },
  loop_fun = function(group, param, g, p) {
    if (is.null(param$grad))
      next
    grad <- param$grad
    
    amsgrad <- group$amsgrad
    weight_decay <- group$weight_decay
    lr <- group$lr
    beta1 <- group$betas[[1]]
    beta2 <- group$betas[[2]]
    eps <- group$eps
    
    # State initialization
    if (length(state(param)) == 0) {
      state(param) <- list()
      state(param)[["step"]] <- torch_scalar_tensor(0, device = param$device)
      # Exponential moving average of gradient values
      state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
      # Exponential moving average of squared gradient values
      state(param)[["exp_avg_sq"]] <- torch::torch_zeros_like(param)
      
      if (amsgrad) {
        state(param)[["max_exp_avg_sqs"]] <- torch::torch_zeros_like(param)
      }
    }
    
    exp_avg      <- state(param)[["exp_avg"]]
    exp_avg_sq   <- state(param)[["exp_avg_sq"]]
    step         <- state(param)[["step"]]
    
    # update step
    step$add_(1)
    
    # Perform stepweight decay
    param$mul_(1 - lr * weight_decay)
    
    # Decay the first and second moment running average coefficient
    exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
    exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value = 1 - beta2)
    
    bias_correction1 <- 1 - beta1^step
    bias_correction2 <- 1 - beta2^step
    
    step_size <- lr / bias_correction1
    
    bias_correction2_sqrt <- sqrt(bias_correction2)
    
    if (amsgrad) {
      # Maintains the maximum of all 2nd moment running avg. till now
      max_exp_avg_sqs <- state(param)[["max_exp_avg_sqs"]]
      torch_maximum_out(max_exp_avg_sqs, exp_avg_sq, max_exp_avg_sqs)
      # Use the max. for normalizing running avg. of gradient
      denom <- (max_exp_avg_sqs$sqrt() / bias_correction2_sqrt)$add_(eps)
    } else {
      denom <- (exp_avg_sq$sqrt() / bias_correction2_sqrt)$add_(eps)
    }
    
    param$addcdiv_(exp_avg, denom, value=-step_size)
  },
  step = function(closure = NULL) {
    private$step_helper(closure, self$loop_fun)
  }
)
