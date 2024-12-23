#' @title Abstract Base Class for LibTorch Optimizers
#' @description
#' Abstract base class for creating optimizers implemented in C++.
#' It is assumed that `self$ptr` is a pointer to the optimizer.
#' Failing to implement this contract will lead to undefined behavior and possibly segfaults when
#' this expected, e.g. by the [`Igniter`] class.
#' @section Sharp Edges:
#' A difference to the `optimizer` class is that the parameters are stored as an unnamed list.
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
    addresses <- sapply(unlist(lapply(param_groups, function(x) x$params)), xptr_address)
    param_groups = lapply(param_groups, function(group) {
      group_param <- sapply(group$params, xptr_address)
      group$params <- match(group_param, addresses)
      group
    })
    if (length(states)) {
      states = lapply(seq(1, length(states) - length(nms) + 1, by = length(nms)), function(i) {
        set_names(states[i:(i + length(nms) - 1)], nms)
      })
    }
    params_with_state = rcpp_ignite_adamw_parameters_with_state(self$ptr)
    params_with_state_addrs = sapply(params_with_state, xptr_address)
    ids = as.character(match(params_with_state_addrs, addresses))
    states = set_names(states, ids)
    # match them with the existing parameters
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
    self$defaults <- list(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
    if (!length(params)) {
      value_error("The `params` must have length > 0.")
    }

    helper = function(pg) {
      params = pg$params
      pg$params = NULL
      opts = c(pg, self$defaults[!(names(self$defaults) %in% names(pg))])
      opts
    }

    if (is.list(params) && is.list(params[[1]]$params)) {
      opts = helper(params[[1]])
      opts$beta1 = opts$betas[1]
      opts$beta2 = opts$betas[2]
      opts$betas = NULL
      self$ptr <- do.call(rcpp_ignite_adamw, c(list(params = params[[1]]$params), opts))
      for (pg in params[-1]) {
        self$add_param_group(pg)
      }
    } else {
      defaults = self$defaults
      defaults$beta1 = defaults$betas[1]
      defaults$beta2 = defaults$betas[2]
      defaults$betas = NULL
      self$ptr <- do.call(rcpp_ignite_adamw, c(list(params = params), defaults))
    }
  },
  state_dict = function() {
    extract_ignite_state_dict(self, rcpp_ignite_adamw_get_states(self$ptr),
      c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"))
  },
  load_state_dict = function(state_dict) {
    if (!is.list(state_dict) || !all(c("param_groups", "state") %in% names(state_dict))) {
      value_error("The `state_dict` must be a list with elements 'param_groups' and 'state'.")
    }
    states = state_dict$state
    prev_states = self$state_dict()$state
    if (!(all(names(prev_states) %in% names(states)))) {
      value_error("To-be loaded state dict is missing states for parameters {paste(setdiff(names(prev_states), names(states)), collapse = ', ')}.")
    }
    walk(as.character(seq_along(prev_states)), function(i) {
      if (!identical(names(states[[i]]), names(prev_states[[i]]))) {
        value_error("The {i}-th state has elements with names {paste0(names(prev_states[[i]]), collapse = ', ')} but got {paste0(names(states[[i]]), collapse = ', ')}.")
      }
    })
    params = unlist(lapply(self$param_groups, function(x) x$params))
    params = params[as.integer(names(states))]
    self$param_groups = state_dict$param_groups
    rcpp_ignite_adamw_set_states(self$ptr, params, unlist(states))
    invisible(self)
  },
  step = function(closure = NULL) {
    loss <- if (!is.null(closure)) {
      with_enable_grad(closure())
    }
    rcpp_ignite_adamw_step(self$ptr)
    return(loss)
  },
  zero_grad = function() {
    rcpp_ignite_adamw_zero_grad(self$ptr)
  },
  add_param_group = function(param_group) {
    params = param_group$params
    # check that params is list of tensors
    if (!is.list(params) || !all(sapply(params, is_torch_tensor))) {
      value_error("The `params` must be a list of tensors.")
    }
    param_group$params = NULL
    # insert defaults into param_group
    param_group = c(param_group, self$defaults[!(names(self$defaults) %in% names(param_group))])
    do.call(assert_adamw_params, param_group)
    param_group$beta1 = param_group$betas[1]
    param_group$beta2 = param_group$betas[2]
    param_group$betas = NULL
    do.call(rcpp_ignite_adamw_add_param_group, c(list(opt = self$ptr, params = params), param_group))
  },
  active = list(
    param_groups = function(rhs) {

      if (!missing(rhs)) {
        prev_param_groups = self$state_dict()$param_groups
        if (!is.list(rhs) && length(rhs) == length(prev_param_groups)) {
          value_error("Parameter groups must be a list of the same length as the number of parameter groups.")
        }

        walk(seq_along(prev_param_groups), function(i) {
          prev_param_group = prev_param_groups[[i]]
          new_param_group = rhs[[i]]
          if (!is_permutation(names(new_param_group), names(prev_param_group))) {
            value_error("Parameter groups must have names {paste0(names(prev_param_group), collapse = ', ')} but got {paste0(names(new_param_group), collapse = ', ')}.")
          }

          if (!identical(prev_param_group$params, new_param_group$params)) {
            value_error("Cannot change the indices of the parameter group, use `$add_param_group()` to add a new parameter group.")
          }

          rcpp_ignite_adamw_set_param_group_options(self$ptr, rhs)
        })
      }
      rcpp_as_list_adamw_param_groups(
        rcpp_ignite_adamw_get_param_groups(self$ptr)
      )
    }
  )
)
is_permutation <- function(vec1, vec2) {
  # Check if lengths are the same
  if (length(vec1) != length(vec2)) {
    return(FALSE)
  }

  # Check if sorted elements are the same
  identical(sort(vec1), sort(vec2))
}
