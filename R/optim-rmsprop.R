#' @include optim.R
NULL

#' RMSprop optimizer
#'
#' Proposed by G. Hinton in his course.
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
#' @param weight_decay optional weight decay penalty. (default: 0)
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
#' @includeRmd man/rmd/optim-note.Rmd note
#' @export
optim_rmsprop <- optimizer(
  "optim_rmsprop",
  initialize = function(params, lr = 1e-2, alpha = 0.99, eps = 1e-8,
                        weight_decay = 0, momentum = 0, centered = FALSE) {
    if (lr < 0) {
      value_error("Invalid learning rate: {lr}")
    }
    if (eps < 0) {
      value_error("Invalid epsilon value: {eps}")
    }
    if (momentum < 0) {
      value_error("Invalid momentum value: {momentum}")
    }
    if (weight_decay < 0) {
      value_error("Invalid weight_decay value: {weight_decay}")
    }
    if (alpha < 0) {
      value_error("Invalid alpha value: {alpha}")
    }

    defaults <- list(
      lr = lr, alpha = alpha, eps = eps, weight_decay = weight_decay,
      momentum = momentum, centered = centered
    )

    super$initialize(params, defaults)
  },
  step = function(closure = NULL) {
    private$step_helper(closure, function(group, param, g, p) {
      grad <- param$grad

      # if (grad$is_sparse) {
      #   runtime_error("RMSprop does not support sparse gradients")
      # }

      # state initialization
      if (length(state(param)) == 0) {
        state(param) <- list()
        state(param)[["step"]] <- 0
        state(param)[["square_avg"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())

        if (group$momentum > 0) {
          state(param)[["momentum_buffer"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
        }

        if (group$centered > 0) {
          state(param)[["grad_avg"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
        }
      }

      square_avg <- state(param)[["square_avg"]]
      alpha <- group[["alpha"]]

      state(param)[["step"]] <- state(param)[["step"]] + 1


      if (group[["weight_decay"]] != 0) {
        grad <- grad$add(p, alpha = group[["weight_decay"]])
      }

      square_avg$mul_(alpha)$addcmul_(grad, grad, value = 1 - alpha)

      if (group[["centered"]]) {
        grad_avg <- state(param)[["grad_avg"]]
        grad_avg$mul_(alpha)$add_(grad, alpha = 1 - alpha)
        avg <- square_avg$addcmul(grad_avg, grad_avg, value = -1)$sqrt_()$add_(group[["eps"]])
      } else {
        avg <- square_avg$sqrt()$add_(group[["eps"]])
      }

      if (group[["momentum"]] > 0) {
        buf <- state(param)[["momentum_buffer"]]
        buf$mul_(group[["momentum"]])$addcdiv_(grad, avg)
        param$add_(buf, alpha = -group[["lr"]])
      } else {
        param$addcdiv_(grad, avg, value = -group[["lr"]])
      }
    })
  }
)
