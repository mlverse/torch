#' @include optim.R
NULL

#' Averaged Stochastic Gradient Descent optimizer
#'
#' Proposed in [Acceleration of stochastic approximation by averaging](https://dl.acm.org/doi/10.1137/0330046)
#'
#' @param params (iterable): iterable of parameters to optimize or lists defining
#'   parameter groups
#' @param lr (float): learning rate
#' @param lambda (float, optional): decay term (default: 1e-4)
#' @param alpha (float, optional): power for eta update (default: 0.75)
#' @param t0 (float, optional): point at which to start averaging (default: 1e6)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#'
#' @includeRmd man/rmd/optim-note.Rmd note
#'
#' @examples
#' \dontrun{
#' optimizer <- optim_asgd(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
#'
#' @export
optim_asgd <- optimizer(
  "optim_asgd",
  initialize = function(params, lr = 1e-2, lambda = 1e-4,
                        alpha = 0.75, t0 = 1e6, weight_decay = 0) {
    if (lr < 0) {
      value_error("Invalid learning rate: {lr}")
    }

    if (weight_decay < 0) {
      value_error("Invalid weight_decay value: {weight_decay}")
    }

    defaults <- list(
      lr = lr, lambda = lambda, alpha = alpha,
      t0 = t0, weight_decay = weight_decay
    )

    super$initialize(params, defaults)
  },
  step = function(closure = NULL) {
    private$step_helper(closure, function(group, param, g, p) {
      grad <- param$grad

      if (length(state(param)) == 0) {
        state(param) <- list()
        state(param)[["step"]] <- 0
        state(param)[["eta"]] <- group[["lr"]]
        state(param)[["mu"]] <- 1
        state(param)[["ax"]] <- torch_zeros_like(param, memory_format = torch_preserve_format())
      }

      state(param)[["step"]] <- state(param)[["step"]] + 1

      if (group[["weight_decay"]] != 0) {
        grad <- grad$add(param, alpha = group$weight_decay)
      }

      # decay term
      param$mul_(1 - group$lambda * state(param)$eta)

      # update parameter
      param$add_(grad, alpha = -state(param)$eta)

      # averaging
      if (state(param)[["mu"]] != 1) {
        state(param)[["mu"]]$add_(param$sub(state(param)[["ax"]])$mul(state(param)[["mu"]]))
      } else {
        state(param)[["ax"]]$copy_(param)
      }

      # update eta and mu
      denominator <- (1 + group[["lambda"]] * group[["lr"]] * state(param)[["step"]])^group[["alpha"]]
      state(param)[["eta"]] <- group[["lr"]] / denominator
      state(param)[["mu"]] <- 1 / max(1, state(param)[["step"]] - group[["t0"]])
    })
  }
)
