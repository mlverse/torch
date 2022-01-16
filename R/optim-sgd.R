#' @include optim.R
NULL

#' SGD optimizer
#'
#' Implements stochastic gradient descent (optionally with momentum).
#' Nesterov momentum is based on the formula from
#' On the importance of initialization and momentum in deep learning.
#'
#' @param params (iterable): iterable of parameters to optimize or dicts defining
#'   parameter groups
#' @param lr (float): learning rate
#' @param momentum (float, optional): momentum factor (default: 0)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' @param dampening (float, optional): dampening for momentum (default: 0)
#' @param nesterov (bool, optional): enables Nesterov momentum (default: FALSE)
#'
#' @section Note:
#'
#' The implementation of SGD with Momentum-Nesterov subtly differs from
#' Sutskever et. al. and implementations in some other frameworks.
#'
#' Considering the specific case of Momentum, the update can be written as
#' \deqn{
#'   \begin{array}{ll}
#' v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
#' p_{t+1} & = p_{t} - \mbox{lr} * v_{t+1},
#' \end{array}
#' }
#'
#' where \eqn{p}, \eqn{g}, \eqn{v} and \eqn{\mu} denote the
#' parameters, gradient, velocity, and momentum respectively.
#'
#' This is in contrast to Sutskever et. al. and
#' other frameworks which employ an update of the form
#'
#' \deqn{
#'   \begin{array}{ll}
#' v_{t+1} & = \mu * v_{t} + \mbox{lr} * g_{t+1}, \\
#' p_{t+1} & = p_{t} - v_{t+1}.
#' \end{array}
#' }
#' The Nesterov version is analogously modified.
#'
#' @includeRmd man/rmd/optim-note.Rmd note
#'
#' @examples
#' \dontrun{
#' optimizer <- optim_sgd(model$parameters(), lr = 0.1, momentum = 0.9)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
#'
#' @export
optim_sgd <- optimizer(
  "optim_sgd",
  initialize = function(params, lr = optim_required(), momentum = 0, dampening = 0,
                        weight_decay = 0, nesterov = FALSE) {
    if (!is_optim_required(lr) && lr < 0) {
      value_error("Invalid learning rate: {lr}")
    }

    if (momentum < 0) {
      value_error("Invalid momentum value: {momentum}")
    }

    if (weight_decay < 0) {
      value_error("Invalid weight_decay value: {weight_decay}")
    }

    if (nesterov && (momentum <= 0 || dampening != 0)) {
      value_error("Nesterov momentum requires a momentum and zero dampening")
    }

    defaults <- list(
      lr = lr, momentum = momentum, dampening = dampening,
      weight_decay = weight_decay, nesterov = nesterov
    )

    super$initialize(params, defaults)
  },
  step = function(closure = NULL) {
    private$step_helper(closure, function(group, param, g, p) {
      weight_decay <- group$weight_decay
      momentum <- group$momentum
      dampening <- group$dampening
      nesterov <- group$nesterov

      d_p <- param$grad

      if (weight_decay != 0) {
        d_p <- d_p$add(p, alpha = weight_decay)
      }
      if (momentum != 0) {
        if (is.null(self$state$get(param)) || !"momentum_buffer" %in% names(self$state$get(param))) {
          buf <- torch_clone(d_p)$detach()
          self$state$set(param, list(momentum_buffer = buf))
        } else {
          buf <- self$state$get(param)$momentum_buffer
          buf$mul_(momentum)$add_(d_p, alpha = 1 - dampening)
        }
        if (nesterov) {
          d_p <- d_p$add(buf, alpha = momentum)
        } else {
          d_p <- buf
        }
      }

      param$add_(d_p, alpha = -group$lr)
    })
  }
)
