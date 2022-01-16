#' @include optim.R
NULL

#' Implements Adam algorithm.
#'
#' It has been proposed in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
#'
#' @param params (iterable): iterable of parameters to optimize or dicts defining
#'   parameter groups
#' @param lr (float, optional): learning rate (default: 1e-3)
#' @param betas (`Tuple[float, float]`, optional): coefficients used for computing
#'   running averages of gradient and its square (default: (0.9, 0.999))
#' @param eps (float, optional): term added to the denominator to improve
#'   numerical stability (default: 1e-8)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' @param amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#'   algorithm from the paper [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
#'   (default: FALSE)
#'
#' @includeRmd man/rmd/optim-note.Rmd note
#'
#' @examples
#' \dontrun{
#' optimizer <- optim_adam(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
#'
#' @export
optim_adam <- optimizer(
  "optim_adam",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
                        weight_decay = 0, amsgrad = FALSE) {
    if (lr < 0) {
      value_error("Invalid learning rate: {lr}")
    }

    if (eps < 0) {
      value_error("Invalid eps: {eps}")
    }

    if (betas[[1]] < 0 || betas[[1]] > 1) {
      value_error("Invalid beta parameter at index 1")
    }

    if (betas[[2]] < 0 || betas[[2]] > 1) {
      value_error("Invalid beta parameter at index 2")
    }

    if (weight_decay < 0) {
      value_error("Invalid weight decay value: {weight_decay}")
    }

    defaults <- list(
      lr = lr, betas = betas, eps = eps, weight_decay = weight_decay,
      amsgrad = amsgrad
    )

    super$initialize(params, defaults)
  },
  step = function(closure = NULL) {
    loop_fun <- function(group, param, g, p) {
      grad <- param$grad

      # if (grad$is_sparse) {
      #   runtime_error("Adam does not support sparse gradients, please consider",
      #                 "SparseAdam instead")
      # }
      amsgrad <- group$amsgrad

      # state initialization
      if (length(state(param)) == 0) {
        state(param) <- list()
        state(param)[["step"]] <- 0
        state(param)[["exp_avg"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
        state(param)[["exp_avg_sq"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
        if (amsgrad) {
          state(param)[["max_exp_avg_sq"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
        }
      }

      exp_avg <- state(param)[["exp_avg"]]
      exp_avg_sq <- state(param)[["exp_avg_sq"]]
      if (amsgrad) {
        max_exp_avg_sq <- state(param)[["max_exp_avg_sq"]]
      }
      beta1 <- group$betas[[1]]
      beta2 <- group$betas[[2]]

      state(param)[["step"]] <- state(param)[["step"]] + 1
      bias_correction1 <- 1 - beta1^state(param)[["step"]]
      bias_correction2 <- 1 - beta2^state(param)[["step"]]

      if (group$weight_decay != 0) {
        grad$add_(p, alpha = group$weight_decay)
      }

      # Decay the first and second moment running average coefficient
      exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
      exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value = 1 - beta2)

      if (amsgrad) {

        # Maintains the maximum of all 2nd moment running avg. till now
        max_exp_avg_sq$set_data(max_exp_avg_sq$max(other = exp_avg_sq))
        # Use the max. for normalizing running avg. of gradient
        denom <- (max_exp_avg_sq$sqrt() / sqrt(bias_correction2))$add_(group$eps)
      } else {
        denom <- (exp_avg_sq$sqrt() / sqrt(bias_correction2))$add_(group$eps)
      }

      step_size <- group$lr / bias_correction1
      param$addcdiv_(exp_avg, denom, value = -step_size)
    }
    private$step_helper(closure, loop_fun)
  }
)
