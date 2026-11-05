#' @title Abstract Base Class for LibTorch Optimizers
#' @description
#' Abstract base class for wrapping LibTorch C++ optimizers.
#' @include optim.R utils-data.R optim-adamw.R RcppExports.R
#' @export
OptimizerIgnite <- R6::R6Class(
  "OptimizerIgnite",
  inherit = Optimizer,
  public = list(
    #' @description
    #' Initializes the optimizer with the specified parameters and defaults.
    #' @param params (`list()`)\cr
    #' Either a list of tensors or a list of parameter groups, each containing the `params` to optimizer
    #' as well as the optimizer options such as the learning rate, weight decay, etc.
    #' @param defaults (`list()`)\cr
    #' A list of default optimizer options.
    initialize = function(params, defaults) {
      do.call(private$.assert_params, defaults)
      self$defaults <- defaults
      if (!is.list(params)) {
        params <- list(params)
      }
      if (!length(params)) {
        value_error("The `params` must have length > 0.")
      }
      helper = function(pg) {
        pg$params <- NULL
        opts <- c(pg, self$defaults[!(names(self$defaults) %in% names(pg))])
        opts
      }

      private$.additional_param_groups <- list(list())

      if (is.list(params) && is.list(params[[1]]$params)) {
        opts <- helper(params[[1]])
        self$ptr <- do.call(private$.optim, c(list(params = params[[1]]$params), opts))
        for (pg in params[-1]) {
          self$add_param_group(pg)
        }
      } else {
        defaults <- self$defaults
        self$ptr <- do.call(private$.optim, c(list(params = params), defaults))
      }
    },
    #' @description
    #' Returns the state dictionary containing the current state of the optimizer.
    #' The returned `list()` contains two lists:
    #' * `param_groups`: The parameter groups of the optimizer (`lr`, ...) as well as to which
    #'   parameters they are applied (`params`, integer indices)
    #' * `state`: The states of the optimizer. The names are the indices of the parameters to which
    #'   they belong, converted to character.
    #' @return (`list()`)
    state_dict = function() {
      extract_ignite_state_dict(self, private$.get_states(self$ptr), private$.state_names)
    },
    #' @description
    #' Loads the state dictionary into the optimizer.
    #' @param state_dict (`list()`)\cr
    #' The state dictionary to load into the optimizer.
    load_state_dict = function(state_dict) {
      if (!is.list(state_dict) || !all(c("param_groups", "state") %in% names(state_dict))) {
        value_error("The `state_dict` must be a list with elements 'param_groups' and 'state'.")
      }
      states <- state_dict$state
      prev_states <- self$state_dict()$state
      if (!(all(names(prev_states) %in% names(states)))) {
        value_error("To-be loaded state dict is missing states for parameters {paste(setdiff(names(prev_states), names(states)), collapse = ', ')}.")
      }
      walk(as.character(seq_along(prev_states)), function(i) {
        if (!identical(names(states[[i]]), names(prev_states[[i]]))) {
          value_error("The {i}-th state has elements with names {paste0(names(prev_states[[i]]), collapse = ', ')} but got {paste0(names(states[[i]]), collapse = ', ')}.")
        }
      })
      params <- unlist(lapply(self$param_groups, function(x) x$params))
      params <- params[as.integer(names(states))]
      self$param_groups <- state_dict$param_groups
      private$.set_states(self$ptr, params, unlist(states))
      invisible(self)
    },
    #' @description
    #' Performs a single optimization step.
    #' @param closure (`function()`)\cr
    #' A closure that conducts the forward pass and returns the loss.
    #' @return (`numeric()`)\cr
    #' The loss.
    step = function(closure = NULL) {
      loss <- if (!is.null(closure)) {
        with_enable_grad(closure())
      }
      rcpp_ignite_optim_step(self$ptr)
      return(loss)
    },
    #' @description
    #' Zeros out the gradients of the parameters.
    zero_grad = function() {
      rcpp_ignite_optim_zero_grad(self$ptr)
    },
    #' @description
    #' Adds a new parameter group to the optimizer.
    #' @param param_group (`list()`)\cr
    #'   A parameter group to add to the optimizer.
    #'   This should contain the `params` to optimize as well as the optimizer options.
    #'   For all options that are not specified, the defaults are used.
    add_param_group = function(param_group) {
      params <- param_group$params
      # check that params is list of tensors
      if (!is.list(params) || !all(sapply(params, is_torch_tensor))) {
        value_error("The `params` must be a list of tensors.")
      }
      param_group$params <- NULL
      # insert defaults into param_group
      param_group <- c(param_group, self$defaults[!(names(self$defaults) %in% names(param_group))])
      do.call(private$.assert_params, param_group)
      do.call(private$.add_param_group, c(list(opt = self$ptr, params = params), param_group))
      private$.additional_param_groups[[length(private$.additional_param_groups) + 1]] <- list()
    }
  ),
  active = list(
    #' @description
    #' The parameter groups of the optimizer.
    param_groups = function(rhs) {
      if (!missing(rhs)) {
        cpp_names <- c("params", private$.config_names)
        prev_param_groups <- self$param_groups
        all_params = unlist(lapply(prev_param_groups, function(x) x$params))
        if (!is.list(rhs) && length(rhs) == length(prev_param_groups)) {
          value_error("Parameter groups must be a list of the same length as the number of parameter groups.")
        }
        walk(seq_along(prev_param_groups), function(i) {
          prev_param_group <- prev_param_groups[[i]]
          new_param_group <- rhs[[i]]
          if (!is_subset(cpp_names, names(new_param_group))) {
            value_error("Parameter groups must include names '{paste0(cpp_names, collapse = ', ')}' but only included '{paste0(names(new_param_group), collapse = ', ')}'.")
          }
          new_param_group_additional <- new_param_group[!(names(new_param_group) %in% cpp_names)]
          private$.additional_param_groups[[i]] <- new_param_group_additional
          param_cmp_value = if (is.integer(new_param_group$params)) {
            all_params[new_param_group$params]
          } else {
            new_param_group$params
          }
          if (!identical(prev_param_group$params, param_cmp_value)) {
            value_error("Cannot change the parameter groups, use `$add_param_group()` to add a new parameter group.")
          }
        })
        # the possible additional param groups are simply ignored
        private$.set_param_group_options(self$ptr, rhs)
      }
      pgs = private$.get_param_groups(self$ptr)
      lapply(seq_along(pgs), function(i) {
        c(pgs[[i]], private$.additional_param_groups[[i]])
      })
    }
  ),
  private = list(
    .additional_param_groups = NULL,
    .optim = function(params, ...) stop("Abstract method"),
    .set_states = function(ptr, params, states) stop("Abstract method"),
    .add_param_group = function(ptr, params, options) stop("Abstract method"),
    .get_states = function(ptr) stop("Abstract method"),
    .assert_params = function(...) stop("Abstract method"),
    .set_param_group_options = function(ptr, options) stop("Abstract method"),
    .get_param_groups = function(ptr) stop("Abstract method")
  )
)

#' @title Abstract Base Class for LibTorch Optimizers
#' @description
#' Abstract base class for wrapping LibTorch C++ optimizers.
#' @inheritParams optimizer
#' @export
#' @include optim.R utils-data.R
optimizer_ignite = function (name = NULL, ..., private = NULL,
  active = NULL, parent_env = parent.frame()) {
  optimizer(
    name = c(name, "optim_ignite"),
    inherit = OptimizerIgnite,
    ...,
    private = private,
    active = active,
    parent_env = parent_env
  )
}

#' @title LibTorch implementation of Adagrad
#' @inherit optim_adagrad  description
#' @section Fields and Methods:
#' See [`OptimizerIgnite`].
#' @inheritParams optim_adagrad
#' @export
#' @include optim-adagrad.R
#' @examples
#' \dontrun{
#' optimizer <- optim_ignite_adagrad(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
optim_ignite_adagrad <- optimizer_ignite(
  "optim_ignite_adagrad",
  initialize = function(params, lr = 1e-2, lr_decay = 0, weight_decay = 0,
    initial_accumulator_value = 0, eps = 1e-10) {
    super$initialize(params, defaults = list(lr = lr, lr_decay = lr_decay, weight_decay = weight_decay, initial_accumulator_value = initial_accumulator_value, eps = eps))
  },
  private = list(
    .optim = function(params, ...) {
      rcpp_ignite_adagrad(params = params, ...)
    },
    .get_states = rcpp_ignite_adagrad_get_states,
    .state_names = c("sum", "step"),
    .config_names = c("lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"),
    .set_states = rcpp_ignite_adagrad_set_states,
    .add_param_group = rcpp_ignite_adagrad_add_param_group,
    .assert_params = assert_adagrad_params,
    .set_param_group_options = rcpp_ignite_adagrad_set_param_group_options,
    .get_param_groups = function(ptr) {
      rcpp_as_list_adagrad_param_groups(rcpp_ignite_optim_get_param_groups(ptr))
    }
  )
)

#' @title LibTorch implementation of RMSprop
#' @inherit optim_rmsprop  description
#' @section Fields and Methods:
#' See [`OptimizerIgnite`].
#' @inheritParams optim_rmsprop
#' @export
#'
#' @include optim-rmsprop.R
#' @examples
#' \dontrun{
#' optimizer <- optim_ignite_rmsprop(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
optim_ignite_rmsprop <- optimizer_ignite(
  "optim_ignite_rmsprop",
  initialize = function(params, lr = 1e-2, alpha = 0.99, eps = 1e-8,
    weight_decay = 0, momentum = 0, centered = FALSE) {
    super$initialize(params, defaults = list(lr = lr, alpha = alpha, eps = eps, weight_decay = weight_decay, momentum = momentum, centered = centered))
  },
  private = list(
    .optim = function(params, ...) {
      rcpp_ignite_rmsprop(params = params, ...)
    },
    .get_states = rcpp_ignite_rmsprop_get_states,
    .state_names = c("grad_avg", "square_avg", "momentum_buffer", "step"),
    .config_names = c("lr", "alpha", "eps", "weight_decay", "momentum", "centered"),
    .set_states = rcpp_ignite_rmsprop_set_states,
    .add_param_group = rcpp_ignite_rmsprop_add_param_group,
    .assert_params = assert_rmsprop_params,
    .set_param_group_options = rcpp_ignite_rmsprop_set_param_group_options,
    .get_param_groups = function(ptr) {
      rcpp_as_list_rmsprop_param_groups(rcpp_ignite_optim_get_param_groups(ptr))
    }
  )
)

#' @title LibTorch implementation of SGD
#' @inherit optim_sgd  description
#' @section Fields and Methods:
#' See [`OptimizerIgnite`].
#' @inheritParams optim_sgd
#' @export
#'
#' @include optim-sgd.R
#' @examples
#' \dontrun{
#' optimizer <- optim_ignite_sgd(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
optim_ignite_sgd <- optimizer_ignite(
  "optim_ignite_sgd",
  initialize = function(params, lr = optim_required(), momentum = 0, dampening = 0,
    weight_decay = 0, nesterov = FALSE) {
    super$initialize(params, defaults = list(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov))
  },
  private = list(
    .optim = function(params, ...) {
      rcpp_ignite_sgd(params = params, ...)
    },
    .get_states = rcpp_ignite_sgd_get_states,
    .state_names = "momentum_buffer",
    .config_names = c("lr", "momentum", "dampening", "weight_decay", "nesterov"),
    .set_states = rcpp_ignite_sgd_set_states,
    .add_param_group = rcpp_ignite_sgd_add_param_group,
    .assert_params = assert_sgd_params,
    .set_param_group_options = rcpp_ignite_sgd_set_param_group_options,
    .get_param_groups = function(ptr) {
      rcpp_as_list_sgd_param_groups(rcpp_ignite_optim_get_param_groups(ptr))
    }
  )
)

#' @title LibTorch implementation of Adam
#' @inherit optim_adam description
#' @section Fields and Methods:
#' See [`OptimizerIgnite`].
#' @inheritParams optim_adam
#' @export
#'
#' @include optim-adam.R
#' @examples
#' \dontrun{
#' optimizer <- optim_ignite_adam(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
optim_ignite_adam <- optimizer_ignite(
  "optim_ignite_adam",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
    weight_decay = 0, amsgrad = FALSE) {
    super$initialize(params, defaults = list(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad))
  },
  private = list(
    .optim = function(params, ...) {
      rcpp_ignite_adam(params = params, ...)
    },
    .get_states = rcpp_ignite_adam_get_states,
    .config_names = c("lr", "betas", "eps", "weight_decay", "amsgrad"),
    .state_names = c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"),
    .set_states = rcpp_ignite_adam_set_states,
    .add_param_group = rcpp_ignite_adam_add_param_group,
    .assert_params = assert_adam_params,
    .set_param_group_options = rcpp_ignite_adam_set_param_group_options,
    .get_param_groups = function(ptr) {
      rcpp_as_list_adam_param_groups(rcpp_ignite_optim_get_param_groups(ptr))
    }
  )
)

#' @title LibTorch implementation of AdamW
#' @inherit optim_adamw description
#' @section Fields and Methods:
#' See [`OptimizerIgnite`].
#' @inheritParams optim_adamw
#' @export
#'
#' @include optim-adamw.R
#' @examples
#' \dontrun{
#' optimizer <- optim_ignite_adamw(model$parameters(), lr = 0.1)
#' optimizer$zero_grad()
#' loss_fn(model(input), target)$backward()
#' optimizer$step()
#' }
optim_ignite_adamw <- optimizer_ignite(
  "optim_ignite_adamw",
  initialize = function(params, lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8,
    weight_decay = 1e-2, amsgrad = FALSE) {
    super$initialize(params, defaults = list(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad))
  },
  private = list(
    .optim = function(params, ...) {
      rcpp_ignite_adamw(params = params, ...)
    },
    .get_states = rcpp_ignite_adamw_get_states,
    .config_names = c("lr", "betas", "eps", "weight_decay", "amsgrad"),
    .state_names = c("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"),
    .set_states = rcpp_ignite_adamw_set_states,
    .add_param_group = rcpp_ignite_adamw_add_param_group,
    .assert_params = assert_adamw_params,
    .set_param_group_options = rcpp_ignite_adamw_set_param_group_options,
    .zero_grad = rcpp_ignite_optim_zero_grad,
    .get_param_groups = function(ptr) {
      rcpp_as_list_adamw_param_groups(rcpp_ignite_optim_get_param_groups(ptr))
    }
  )
)

is_subset <- function(vec1, vec2) {
  all(vec1 %in% vec2)
}

is_permutation <- function(vec1, vec2) {
  # Check if lengths are the same
  if (length(vec1) != length(vec2)) {
    return(FALSE)
  }
  # Check if sorted elements are the same
  identical(sort(vec1), sort(vec2))
}

extract_ignite_state_dict <- function(self, states, nms) {
    # the param_groups actually contain the parameters that are optimized.
    # But we don't want to return them as part of the state dict.
    # Therefore, we unlist all the parameters and store the indices in the state dict.
    param_groups <- self$param_groups
    # Keep tensor R objects alive so their pyobj_slot cache (used by
    # xptr_address) is not invalidated by R_RunPendingFinalizers() inside
    # operator_sexp_tensor when we later call parameters_with_state.
    all_params <- unlist(lapply(param_groups, function(x) x$params))
    addresses <- sapply(all_params, xptr_address)
    param_groups = lapply(param_groups, function(group) {
      group_param <- sapply(group$params, xptr_address)
      group$params <- match(group_param, addresses)
      group
    })
    if (length(states)) {
      states <- lapply(seq(1, length(states) - length(nms) + 1, by = length(nms)), function(i) {
        set_names(states[i:(i + length(nms) - 1)], as.character(nms))
      })
    }
    params_with_state <- rcpp_ignite_optim_parameters_with_state(self$ptr)
    params_with_state_addrs <- sapply(params_with_state, xptr_address)
    ids <- as.character(match(params_with_state_addrs, addresses))
    states <- set_names(states, ids)
    # match them with the existing parameters
    list(
      param_groups = param_groups,
      state = states
    )
}
