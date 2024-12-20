#' @title Abstract Base Class for LibTorch Optimizers
#' @description
#' Abstract base class for creating optimizers implemented in C++.
#' It is assumed that `self$ptr` is a pointer to the optimizer.
#' Failing to implement this contract will lead to undefined behavior and possibly segfaults when
#' this expected, e.g. by the [`Igniter`] class.
#' @inheritParams torch::optimizer
#' @section State Dict:
#' The `$state_dict()` method returns a list with two elements:
#' - `param_groups`: A list of parameter groups.
#'   Each parameter group contains a field:
#'   - `params`: An integer vector indicating the indices of the parameters in the optimizer.
#'   - other arbitrary fields such as `lr`, `weight_decay`, etc.
#' - `states`: A list of optimizer states. The length of this list is the same as the number of parameters.
#'    The structure of the optimizer states is specific to the optimizer.
#' @section Loading State Dict:
#' The `$load_state_dict()` method loads the state dict.
#' @export
optimizer_ignite = function (name = NULL, ..., private = NULL,
  active = NULL, parent_env = parent.frame()) {
  optimizer(
    name = c(name, "optim_ignite"),
    inherit = NULL,
    ...,
    private = private,
    active = active,
    parent_env = parent_env
  )
}

extract_ignite_state_dict = function(self, states, nms) {
    # the param_groups actually contain the parameters that are optimized.
    # But we don't want to return them as part of the state dict.
    # Therefore, we unlist all the parameters and store the indices in the state dict.
    param_groups = self$param_groups
    addresses <- sapply(unlist(lapply(param_groups, function(x) x$params)), torch:::xptr_address)
    param_groups = lapply(param_groups, function(group) {
      group_param <- sapply(group$params, torch:::xptr_address)
      group$params <- match(group_param, addresses)
      group
    })
    states = lapply(seq(1, length(states) - length(nms) + 1, by = length(nms)), function(i) {
      current_state = states[i:(i + length(nms) - 1)]
      # the parameter has no state
      set_names(states[i:(i + length(nms) - 1)], nms)
    })
    if (is.null(states)) states <- set_names(list(), character())

    list(
      param_groups = param_groups,
      state = states
    )
}

#' @export
#' @title SGD Optimizer as implemented in LibTorch
#' @inheritParams torch::optim_adam
#' @export
optim_ignite_adamw <- optimizer_ignite(
  "optim_ignite_adamw",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
                       weight_decay = 1e-2, amsgrad = FALSE) {
    assert_adamw_params(lr, betas, eps, weight_decay, amsgrad)
    self$ptr <- rcpp_ignite_adamw(params = params, lr = lr, beta1 = betas[1], beta2 = betas[2],
      eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
  },
  state_dict = function() {
    extract_ignite_state_dict(self, rcpp_ignite_adamw_get_states(self$ptr),
      c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"))
  },
  load_state_dict = function(state_dict) {
    self$param_groups = state_dict$param_groups
    states = unlist(state_dict$state)
    rcpp_ignite_adamw_set_states(self$ptr, states)
    invisible(self)
  },
  step = function(closure = NULL) {
    loss = if (!is.null(closure)) {
      with_enable_grad(closure())
    }
    rcpp_ignite_adamw_step(self$ptr)
    return(loss)
  },
  zero_grad = function() {
    rcpp_ignite_adamw_zero_grad(self$ptr)
  },
  active = list(
    # TODO: Add the params as an integer vector.
    param_groups = function(rhs) {
      if (!missing(rhs)) {
        # TODO: Check that params are not changed.
        rcpp_ignite_adamw_set_param_group_options(self$ptr, rhs)
      }
      rcpp_as_list_adamw_param_groups(
        rcpp_ignite_adamw_get_param_groups(self$ptr)
      )
    }
  )
)
