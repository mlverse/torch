#' @include optim.R
NULL

#' Adagrad optimizer
#'
#' Proposed in [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)
#'
#' @param params (iterable): list of parameters to optimize or list parameter groups
#' @param lr (float, optional): learning rate (default: 1e-2)
#' @param lr_decay (float, optional): learning rate decay (default: 0)
#' @param weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#' @param eps (float, optional): term added to the denominator to improve
#'   numerical stability (default: 1e-10)
#' @param initial_accumulator_value the initial value for the accumulator. (default: 0)
#'
#' Adagrad is an especially good optimizer for sparse data.
#' It individually modifies learning rate for every single parameter,
#' dividing the original learning rate value by sum of the squares of the gradients.
#' It causes that the rarely occurring features get greater learning rates.
#' The main downside of this method is the fact that learning rate may be
#' getting small too fast, so that at some point a model cannot learn anymore.
#'
#' @note
#' Update rule:
#' \deqn{
#' \theta_{t+1} = \theta_{t} - \frac{\eta }{\sqrt{G_{t} + \epsilon}} \odot g_{t}
#' }
#' The equation above and some remarks quoted
#' after [*An overview of gradient descent optimization algorithms*](https://ruder.io/optimizing-gradient-descent/index.html#adagrad)
#' by Sebastian Ruder.
#' @includeRmd man/rmd/optim-note.Rmd note
#' @export
optim_adagrad <- optimizer(
  "optim_adagrad",
  initialize = function(params, lr = 1e-2, lr_decay = 0, weight_decay = 0,
                        initial_accumulator_value = 0, eps = 1e-10) {
    if (lr < 0) {
      value_error("Invalid learning rate: {lr)}")
    }

    if (lr_decay < 0) {
      value_error("Invalid lr_decay value: {lr_decay}")
    }

    if (weight_decay < 0) {
      value_error("Invalid weight_decay value: {weight_decay}")
    }

    if (initial_accumulator_value < 0) {
      value_error("Invalid initial_accumulator_value value: {initial_accumulator_value}")
    }

    if (eps < 0) {
      value_error("Invalid epsilon value: {eps}")
    }

    defaults <- list(
      lr = lr, lr_decay = lr_decay, weight_decay = weight_decay,
      initial_accumulator_value = initial_accumulator_value,
      eps = eps
    )
    super$initialize(params, defaults)

    for (group in self$param_groups) {
      for (p in seq_along(group$params)) {
        param <- group$params[[p]]
        state(param) <- list()
        state(param)[["step"]] <- 0
        state(param)[["sum"]] <- torch_full_like(
          param,
          initial_accumulator_value,
          memory_format = torch_preserve_format()
        )
      }
    }
  },

  # It's implemeneted in PyTorch, but it's not necessary at the moment
  # share_memory = function(){
  #   for (group in self$param_groups){
  #     for (p in seq_along(group$params)) {
  #       param <- group$params[[p]]
  #       state(param)[['sum']]$share_memory_()
  #     }
  #   }
  # },

  step = function(closure = NULL) {
    private$step_helper(closure, function(group, param, g, p) {
      state(param)[["step"]] <- state(param)[["step"]] + 1

      grad <- param$grad
      state_sum <- state(param)[["sum"]]
      state_step <- state(param)[["step"]]

      if (group$weight_decay != 0) {
        # if (grad$is_sparse) {
        #   runtime_error("weight_decay option is not compatible with sparse gradients")
        # }
        grad <- grad$add(param, alpha = group$weight_decay)
      }

      clr <- group$lr / (1 + (state(param)[["step"]] - 1) * group$lr_decay)

      # Sparse tensors handling will be added in future
      # if (grad$is_sparse) {
      #   grad <- grad$coalesce()
      #   grad_indices <- grad$`_indices`()
      #   grad_values <-  grad$`_values`()
      #   size <- grad$size()

      # state_sum$add_(`_make_sparse`(grad, grad_indices, grad_values.pow(2)))
      # std <- state(param)[['sum']]$sparse_mask(grad)
      # std_values <- std$`_values()`$sqrt_()$add_(group$eps)
      # param$add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
      # } else {

      state(param)[["sum"]]$addcmul_(grad, grad, value = 1)
      std <- state(param)[["sum"]]$sqrt()$add_(group$eps)
      param$addcdiv_(grad, std, value = -clr)
    })
  }
)
