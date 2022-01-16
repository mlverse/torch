#' Dummy value indicating a required value.
#'
#' export
optim_required <- function() {
  structure(list(), class = "optim_required")
}

is_optim_required <- function(x) {
  inherits(x, "optim_required")
}

#' Checks if the object is a torch optimizer
#'
#' @param x object to check
#'
#' @export
is_optimizer <- function(x) {
  inherits(x, "torch_Optimizer")
}

Optimizer <- R6::R6Class(
  "torch_Optimizer",
  lock_objects = FALSE,
  public = list(
    initialize = function(params, defaults) {
      self$defaults <- defaults
      self$state <- list()
      self$param_groups <- list()

      if (is_torch_tensor(params)) {
        param_groups <- list(list(params = list(params)))
      } else if (is.list(params) && is_torch_tensor(params[[1]])) {
        param_groups <- list(list(params = params))
      } else if (rlang::is_named(params[[1]])) {
        param_groups <- params
      } else {
        value_error("Wrong parameters specification.")
      }

      for (p in param_groups) {
        self$add_param_group(p)
      }

      self$state <- State$new()
    },
    add_param_group = function(param_group) {
      if (!rlang::is_named(param_group)) {
        value_error("param group is not named")
      }

      params <- param_group$params
      if (is_torch_tensor(params)) {
        param_group$params <- list(params)
      }

      for (param in param_group$params) {
        if (!is_torch_tensor(param)) {
          value_error(
            "optimizer can only optimize Tensors, ",
            "but one of the params is {class(param)}"
          )
        }

        if (!param$is_leaf) {
          value_error("can't optimize a non-leaf Tensor")
        }
      }


      for (nm in names(self$defaults)) {
        if (is_optim_required(self$defaults[[nm]]) && !nm %in% names(param_group)) {
          value_error(
            "parameter group didn't specify a value of required ",
            "optimization parameter {nm}"
          )
        } else if (!nm %in% names(param_group)) {
          param_group[[nm]] <- self$defaults[[nm]]
        }
      }

      # TODO: check for duplicated parameters

      self$param_groups <- append(self$param_groups, list(param_group))
    },
    zero_grad = function() {
      for (group in self$param_groups) {
        for (p in group$params) {
          grad <- p$grad
          if (!is_undefined_tensor(grad)) {
            grad$detach_()
            grad$zero_()
          }
        }
      }
    },
    state_dict = function() {
      parameters <- unlist(lapply(self$param_groups, function(x) x$params))
      parameters <- lapply(parameters, xptr_address)

      state_dict <- self$state$map
      names(state_dict) <- match(names(self$state$map), parameters)

      param_groups <- self$param_groups
      param_groups <- lapply(param_groups, function(x) {
        group_param <- lapply(x$params, xptr_address)
        x$params <- match(group_param, parameters)
        x
      })

      list(
        param_groups = param_groups,
        state = state_dict
      )
    },
    load_state_dict = function(state_dict) {

      # validate the state dict
      if (!length(self$param_groups) == length(state_dict$param_groups)) {
        value_error("Loaded state dict has a different number of parameter groups")
      }

      for (i in seq_along(self$param_groups)) {
        if (!length(self$param_groups[[i]]$params) == length(state_dict$param_groups[[i]]$params)) {
          value_error("Loaded state dict has contains a parameter group that doesn't match the size of optimizers group.")
        }
      }

      parameters <- unlist(lapply(self$param_groups, function(x) x$params))

      # update state
      for (o in names(state_dict$state)) {
        index <- as.integer(o)
        value <- state_dict$state[[o]]
        self$state$set(parameters[[index]], value)
      }

      invisible(self)
    }
  ),
  private = list(
    step_helper = function(closure, loop_fun) {
      # a general template for most of the optimizer step function
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

            if (is.null(param$grad) || is_undefined_tensor(param$grad)) {
              next
            }

            loop_fun(group, param, g, p)
          }
        }
        loss
      })
    }
  )
)

State <- R6::R6Class(
  "State",
  lock_objects = FALSE,
  public = list(
    initialize = function() {
      self$map <- list()
    },
    set = function(key, value) {
      self$map[[xptr_address(key)]] <- value
    },
    get = function(key) {
      self$map[[xptr_address(key)]]
    },
    keys = function() {
      names(self$map)
    }
  )
)

state <- function(self) {
  rlang::env_get(rlang::caller_env(), "self", default = NULL, inherit = TRUE)$state$get(self)
}

`state<-` <- function(self, value) {
  rlang::env_get(rlang::caller_env(), "self", default = NULL, inherit = TRUE)$state$set(self, value)
  self
}

#' Creates a custom optimizer
#'
#' When implementing custom optimizers you will usually need to implement
#' the `initialize` and `step` methods. See the example section below
#' for a full example.
#'
#' @param name (optional) name of the optimizer
#' @param inherit (optional) you can inherit from other optimizers to re-use
#'   some methods.
#' @param ... Pass any number of fields or methods. You should at least define
#'   the `initialize` and `step` methods. See the examples section.
#' @param private (optional) a list of private methods for the optimizer.
#' @param active (optional) a list of active methods for the optimizer.
#' @param parent_env used to capture the right environment to define the class.
#'   The default is fine for most situations.
#'
#' @examples
#'
#' # In this example we will create a custom optimizer
#' # that's just a simplified version of the `optim_sgd` function.
#'
#' optim_sgd2 <- optimizer(
#'   initialize = function(params, learning_rate) {
#'     defaults <- list(
#'       learning_rate = learning_rate
#'     )
#'     super$initialize(params, defaults)
#'   },
#'   step = function() {
#'     with_no_grad({
#'       for (g in seq_along(self$param_groups)) {
#'         group <- self$param_groups[[g]]
#'         for (p in seq_along(group$params)) {
#'           param <- group$params[[p]]
#'
#'           if (is.null(param$grad) || is_undefined_tensor(param$grad)) {
#'             next
#'           }
#'
#'           param$add_(param$grad, alpha = -group$learning_rate)
#'         }
#'       }
#'     })
#'   }
#' )
#'
#' x <- torch_randn(1, requires_grad = TRUE)
#' opt <- optim_sgd2(x, learning_rate = 0.1)
#' for (i in 1:100) {
#'   opt$zero_grad()
#'   y <- x^2
#'   y$backward()
#'   opt$step()
#' }
#' all.equal(x$item(), 0, tolerance = 1e-9)
#' @includeRmd man/rmd/optim-note.Rmd note
#' @export
optimizer <- function(name = NULL, inherit = Optimizer, ...,
                      private = NULL, active = NULL,
                      parent_env = parent.frame()) {
  create_class(
    name = name,
    inherit = inherit,
    ...,
    private = private,
    active = active,
    parent_env = parent_env,
    attr_name = "Optimizer"
  )
}
