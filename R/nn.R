#' @include utils-data.R
NULL

get_inherited_classes <- function(inherit) {
  inherit_class <- inherit$public_fields$.classes
  # Filter out classes that we eventually add in our normal flow.
  inherit_class <- inherit_class[inherit_class != "nn_Module"]
  inherit_class <- inherit_class[inherit_class != "R6ClassGenerator"]
  inherit_class <- inherit_class[inherit_class != "nn_module_generator"]
  inherit_class <- inherit_class[inherit_class != "nn_module"]
  inherit_class <- inherit_class[!duplicated(inherit_class, fromLast = TRUE)]
  inherit_class
}

nn_Module <- R6::R6Class(
  classname = "nn_Module",
  lock_objects = FALSE,
  public = list(
    initialize = function() {},
    forward = function(...) {
      not_implemented_error("Forward method is not implemented")
    },
    add_module = function(name, module) {
      self$register_module(name, module)
    },
    register_module = function(name, module) {
      if (is.numeric(name)) {
        name <- as.character(name)
      }

      if (!inherits(module, "nn_module")) {
        cli_abort("Expected {.cls nn_module} but got object of type {.cls {class(module)}}")
      }

      if (inherits(module, "nn_module_generator")) {
        cli_abort("Module {.str {name}} must be initialized")
      }

      private$modules_[[name]] <- module
    },
    register_parameter = function(name, param) {
      private$parameters_[[name]] <- param
    },
    register_buffer = function(name, tensor, persistent = TRUE) {
      private$buffers_[[name]] <- tensor

      if (persistent) {
        private$non_persistent_buffers_ <- private$non_persistent_buffers_[
          private$non_persistent_buffers_ != name
        ]
      } else {
        private$non_persistent_buffers_ <- unique(c(
          private$non_persistent_buffers_,
          name
        ))
      }
    },
    train = function(mode = TRUE) {
      self$training <- mode
      lapply(private$modules_, function(m) m$train(mode))
      invisible(create_nn_module_callable(self))
    },
    eval = function() {
      self$train(FALSE)
    },
    .apply = function(fn) {
      for (module in private$modules_) {
        module$.apply(fn)
      }

      for (param_name in names(private$parameters_)) {
        param <- private$parameters_[[param_name]]

        if (!is.null(param)) {
          # Tensors stored in modules are graph leaves, and we don't want to
          # track autograd history of `param_applied`, so we have to use
          # `with torch.no_grad():`
          with_no_grad({
            param_applied <- fn(param)
          })
          private$parameters_[[param_name]] <-
            nn_parameter(param_applied, param$requires_grad)
        }

        if (!is.null(param) && !is_undefined_tensor(param$grad)) {
          with_no_grad({
            grad_applied <- fn(param$grad)
          })
          grad_applied$requires_grad_(param$grad$requires_grad)
          private$parameters_[[param_name]]$set_grad_(grad_applied)
        }
      }

      for (buf_name in names(private$buffers_)) {
        buf <- private$buffers_[[buf_name]]
        if (!is.null(buf)) {
          private$buffers_[[buf_name]] <- fn(buf)
        }
      }

      invisible(create_nn_module_callable(self))
    },
    cuda = function(device = NULL) {
      self$.apply(function(x) x$cuda())
    },
    cpu = function() {
      self$.apply(function(x) x$cpu())
    },
    to = function(dtype = NULL, device = NULL, tensor = NULL, non_blocking = FALSE, copy = FALSE,
                  memory_format = torch_preserve_format()) {
      if (!is.null(dtype)) {
        if (!dtype$is_floating_point) {
          value_error("nn.Module.to only accepts floating point '
                      'dtypes, but got desired dtype {dtype}")
        }
      }

      self$.apply(function(x) {
        if (x$is_floating_point()) {
          x$to(dtype, device, tensor, non_blocking, copy, memory_format)
        } else {
          x$to(
            device = device, non_blocking = non_blocking, copy = copy,
            memory_format = memory_format
          )
        }
      })
    },
    print = function() {
      print_nn_module(self, private)
    },
    .save_to_state_dict = function(prefix, keepvars) {
      out <- list()

      for (param_name in names(private$parameters_)) {
        param <- private$parameters_[[param_name]]
        if (!is.null(param)) {
          if (!keepvars) {
            param$detach
          }
          out[[paste0(prefix, param_name)]] <- keepvars_or_detach(param, keepvars)
        }
      }

      for (buf_name in names(private$buffers_)) {
        buf <- private$buffers_[[buf_name]]
        if (!is.null(buf) && !(buf_name %in% private$non_persistent_buffers_)) {
          out[[paste0(prefix, buf_name)]] <- keepvars_or_detach(buf, keepvars)
        }
      }

      out
    },
    state_dict = function(prefix = "", keepvars = FALSE) {
      out <- list()
      out <- c(out, self$.save_to_state_dict(prefix, keepvars))

      for (module_name in names(private$modules_)) {
        module <- private$modules_[[module_name]]
        if (!is.null(module)) {
          out <- c(out, module$state_dict(
            prefix = paste0(prefix, module_name, "."),
            keepvars = keepvars
          ))
        }
      }

      out
    },
    .load_from_state_dict = function(state_dict, prefix) {
      persistent_buffers <- private$buffers_[!names(private$buffers_) %in% private$non_persistent_buffers_]
      local_name_params <- c(private$parameters_, persistent_buffers)
      local_state <- local_name_params[!sapply(local_name_params, is.null)]

      for (name in names(local_state)) {
        key <- paste0(prefix, name)
        if (key %in% names(state_dict)) {
          input_param <- state_dict[[key]]
          param <- local_state[[name]]
          if (!self$..refer_to_state_dict..) {
            with_no_grad({
              param$copy_(input_param)
            })
          } else {

            # setting requires grad is ignored if param is not a valid pointer
            # be careful!
            if (!is_null_external_pointer(param)) {
              input_param$requires_grad_(param$requires_grad)
            }

            if (name %in% names(persistent_buffers)) {
              private$buffers_[[name]] <- input_param
            } else {
              private$parameters_[[name]] <- input_param
            }
          }
        } else {
          value_error("Could not find {key} in the state_dict.")
        }
      }
    },
    load_state_dict = function(state_dict, ..., .refer_to_state_dict = FALSE) {
      # by default the state dict parameter values are copied into the parameters
      # of the modules. with `.refer_to_state_dict` you can make the parameters
      # refer to the tensors in the state dict. USE WITH CAUTION as it's easy to
      # mess up and link the parametrs of two models that way. This is useful when
      # you want to initialize the model with the state dict values and will dispose
      # of the state dict rightly after.
      load <- function(module, state_dict, prefix = "") {
        module$..refer_to_state_dict.. <- .refer_to_state_dict
        module$.load_from_state_dict(state_dict, prefix)
        for (nm in names(module$.__enclos_env__$private$modules_)) {
          child <- module$.__enclos_env__$private$modules_[[nm]]
          if (!is.null(child)) {
            load(child, state_dict, prefix = paste0(prefix, nm, "."))
          }
        }
      }

      load(self, state_dict)

      invisible(create_nn_module_callable(self))
    },
    zero_grad = function() {
      for (p in self$parameters) {
        if (!is_undefined_tensor(p$grad)) {
          p$grad$detach_()
          p$grad$zero_()
        }
      }
    },
    apply = function(fn) {
      for (module in private$modules_) {
        module$apply(fn)
      }
      fn(self)
      invisible(create_nn_module_callable(self))
    },
    named_parameters = function(recursive = TRUE) {
      if (recursive) {
        self$parameters
      } else {
        private$parameters_
      }
    },
    named_buffers = function(recursive = TRUE) {
      if (recursive) {
        self$buffers
      } else {
        private$buffers_
      }
    },
    .replace_values_from_table = function(table) {
      for (i in seq_along(private$modules_)) {
        module <- private$modules_[[i]]
        private$modules_[[i]] <- table[[rlang::obj_address(module)]] %||% module
      }

      lapply(private$modules_, function(x) x$.replace_values_from_table(table))

      for (i in seq_along(private$parameters_)) {
        par <- private$parameters_[[i]]
        # par or buf might not be available in `table` if, for some reason they
        # have already been replaced. This happens for example, when a module
        # has the same layer twice. this also applies for modules, they might be duplicated
        private$parameters_[[i]] <- table[[xptr_address(par)]] %||% par
      }
      for (i in seq_along(private$buffers_)) {
        buf <- private$buffers_[[i]]
        private$buffers_[[i]] <- table[[xptr_address(buf)]] %||% buf
      }
    }
  ),
  private = list(
    training_ = TRUE,
    parameters_ = list(),
    buffers_ = list(),
    modules_ = list(),
    non_persistent_buffers_ = character()
  ),
  active = list(
    training = function(rhs) {
      if (!missing(rhs)) {
        if (!(length(rhs) == 1 && is.logical(rhs) && !is.na(rhs))) {
          value_error("Field `training` must be a logical flag.")
        }
        private$training_ = rhs
      }
      private$training_
    },
    parameters = function(value, recursive = TRUE) {
      if (!missing(value)) {
        runtime_error(
          "It's not possible to modify the parameters list.\n",
          " You can modify the parameter in-place or use",
          " `module$parameter_name <- new_value`"
        )
      }

      pars <- lapply(private$modules_, function(x) x$parameters)
      pars <- append(pars, self$named_parameters(recursive = FALSE))
      pars <- unlist(pars, recursive = TRUE, use.names = TRUE)
      pars <- pars[!duplicated(pars)] # unique doesn't preserve the names

      pars
    },
    buffers = function(value) {
      if (!missing(value)) {
        runtime_error(
          "It's not possible to modify the buffers list.\n",
          " You can modify the parameter in-place or use",
          " `module$parameter_name <- new_value`"
        )
      }

      bufs <- lapply(private$modules_, function(x) x$buffers)
      bufs <- append(bufs, self$named_buffers(recursive = FALSE))
      bufs <- unlist(bufs, recursive = TRUE, use.names = TRUE)
      bufs <- bufs[!duplicated(bufs)] # unique doesn't preserve the names

      bufs
    },
    modules = function(value) {
      if (!missing(value)) {
        runtime_error(
          "It's not possible to modify the modules list.\n",
          " You can modify the modules in-place"
        )
      }

      modules <- lapply(private$modules_, function(x) x$modules)
      # the self instance is an nn_Module, not nn_module
      modules <- append(create_nn_module_callable(self), modules)
      modules <- unlist(modules)

      # to check if modules are iddentical we need to compare the
      # R6 instances.
      module_instances <- lapply(modules, function(x) attr(x, "module"))
      modules <- modules[!duplicated(module_instances)]

      modules
    },
    children = function(value) {
      if (!missing(value)) {
        runtime_error(
          "It's not possible to modify the children list.\n",
          " You can modify the modules in-place"
        )
      }

      private$modules_
    }
  )
)


#' Creates an `nn_parameter`
#'
#' Indicates to nn_module that `x` is a parameter
#'
#' @param x the tensor that you want to indicate as parameter
#' @param requires_grad whether this parameter should  have
#'   `requires_grad = TRUE`
#'
#' @export
nn_parameter <- function(x, requires_grad = TRUE) {
  if (!is_torch_tensor(x)) {
    type_error("`x` must be a tensor.")
  }
  x$requires_grad_(requires_grad)
  class(x) <- c(class(x), "nn_parameter")
  x
}

#' Checks if an object is a nn_parameter
#'
#' @param x the object to check
#'
#' @export
is_nn_parameter <- function(x) {
  inherits(x, "nn_parameter")
}

#' Creates a nn_buffer
#'
#' Indicates that a tensor is a buffer in a nn_module
#'
#' @param x the tensor that will be converted to nn_buffer
#' @param persistent whether the buffer should be persistent or not.
#'
#' @export
nn_buffer <- function(x, persistent = TRUE) {
  class(x) <- c(class(x), "nn_buffer")
  attr(x, "persistent") <- persistent
  x
}

#' Checks if the object is a nn_buffer
#'
#' @param x object to check
#'
#' @export
is_nn_buffer <- function(x) {
  inherits(x, "nn_buffer")
}

#' Checks if the object is an nn_module
#'
#' @param x object to check
#'
#' @export
is_nn_module <- function(x) {
  inherits(x, "nn_module") && !inherits(x, "nn_module_generator")
}

#' Base class for all neural network modules.
#'
#' Your models should also subclass this class.
#'
#' Modules can also contain other Modules, allowing to nest them in a tree
#' structure. You can assign the submodules as regular attributes.
#'
#' You are expected to implement the `initialize` and the `forward` to create a
#' new `nn_module`.
#'
#' @section Initialize:
#'
#' The initialize function will be called whenever a new instance of the `nn_module`
#' is created. We use the initialize functions to define submodules and parameters
#' of the module. For example:
#'
#' ```
#' initialize = function(input_size, output_size) {
#'    self$conv1 <- nn_conv2d(input_size, output_size, 5)
#'    self$conv2 <- nn_conv2d(output_size, output_size, 5)
#' }
#' ```
#'
#' The initialize function can have any number of parameters. All objects
#' assigned to `self$` will be available for other methods that you implement.
#' Tensors wrapped with [nn_parameter()] or [nn_buffer()] and submodules are
#' automatically tracked when assigned to `self$`.
#'
#' The initialize function is optional if the module you are defining doesn't
#' have weights, submodules or buffers.
#'
#' @section Forward:
#'
#' The forward method is called whenever an instance of `nn_module` is called.
#' This is usually used to implement the computation that the module does with
#' the weights ad submodules defined in the `initialize` function.
#'
#' For example:
#'
#' ```
#' forward = function(input) {
#'    input <- self$conv1(input)
#'    input <- nnf_relu(input)
#'    input <- self$conv2(input)
#'    input <- nnf_relu(input)
#'    input
#'  }
#' ```
#'
#' The `forward` function can use the `self$training` attribute to make different
#' computations depending wether the model is training or not, for example if you
#' were implementing the dropout module.
#'
#' @section Cloning:
#' To finalize the cloning of a module, you can define a private `finalize_deep_clone()` method.
#' This method is called on the cloned object when deep-cloning a module, after all the modules, parameters and
#' buffers were already cloned.
#'
#' @param classname an optional name for the module
#' @param inherit an optional module to inherit from
#' @param ... methods implementation
#' @param private passed to [R6::R6Class()].
#' @param active passed to [R6::R6Class()].
#' @param parent_env passed to [R6::R6Class()].
#'
#' @examples
#' model <- nn_module(
#'   initialize = function() {
#'     self$conv1 <- nn_conv2d(1, 20, 5)
#'     self$conv2 <- nn_conv2d(20, 20, 5)
#'   },
#'   forward = function(input) {
#'     input <- self$conv1(input)
#'     input <- nnf_relu(input)
#'     input <- self$conv2(input)
#'     input <- nnf_relu(input)
#'     input
#'   }
#' )
#' @export
nn_module <- function(classname = NULL, inherit = nn_Module, ...,
                      private = NULL, active = NULL,
                      parent_env = parent.frame()) {
  if (inherits(inherit, "nn_module")) {
    inherit <- attr(inherit, "module")
  }

  e <- new.env(parent = parent_env)
  e$inherit <- inherit

  inherit_class <- get_inherited_classes(inherit)
  classes <- c(classname, inherit_class, "nn_module")

  Module <- R6::R6Class(
    classname = classname,
    inherit = inherit,
    lock_objects = FALSE,
    public = list(
      .classes = classes,
      ...
    ),
    private = private,
    active = active,
    parent_env = e
  )

  # We can't patch $clone() in nn_Moudle, as the inheriting classes will not inherit the patched $clone() method

  # as R6's clone method is quite restrictive, we here assign the original public $clone() method to the private
  # field $.__clone_r6__ and create a new patched $clone() method
  # This circumvents some restrictions in R6, see e.g. this discussion: https://github.com/r-lib/R6/issues/179
  Module$set("private", ".__clone_r6__", nn_Module$public_methods$clone, overwrite = TRUE)
  # because the clone method is encapsulated (happens during $new()), it cannot access torch's internal functions and we
  # hence need to call into torch's public API
  # We even explicitely annotate the namespace, as it might otherwise be the case, that other packages
  # define their own custom modules and (accidentally) a clone_module function which would lead to this clone_module
  # function being found before torch's clone_module
  Module$set("public", "clone", overwrite = TRUE, value = function(deep = FALSE, ..., replace_values = TRUE) {
      torch::clone_module(self, deep = deep, ..., replace_values = replace_values)
    }
  )


  init <- get_init(Module)

  fun <- rlang::new_function(
    args = rlang::fn_fmls(init),
    body = rlang::expr({
      instance <- Module$new(!!!rlang::fn_fmls_syms(init))
      create_nn_module_callable(instance)
    })
  )
  attr(fun, "class") <- c(classes, "nn_module_generator")
  attr(fun, "module") <- Module
  fun
}

create_nn_module_callable <- function(instance) {
  f <- instance$forward

  attr(f, "class") <- instance$.classes
  attr(f, "module") <- instance
  f
}

#' @export
`$.nn_module` <- function(x, y) {
  module <- attr(x, "module")
  do.call("$", args = list(module, y))
}

#' @export
`[[.nn_module` <- function(x, y) {
  module <- attr(x, "module")
  do.call("[[", args = list(module, y))
}

#' @export
`$.nn_Module` <- function(x, y) {
  x[[y]]
}

#' @export
`[[.nn_Module` <- function(x, y) {
  if (y == ".__enclos_env__") {
    return(NextMethod())
  }

  if (is.numeric(y)) {
    return(x[[".__enclos_env__"]][["private"]][["modules_"]][[y]])
  }

  pars <- x[[".__enclos_env__"]][["private"]][["parameters_"]]
  if (!is.null(pars)) {
    o <- pars[[y]]
    if (!is.null(o)) {
      return(o)
    }
  }

  bufs <- x[[".__enclos_env__"]][["private"]][["buffers_"]]
  if (!is.null(bufs)) {
    o <- bufs[[y]]
    if (!is.null(o)) {
      return(o)
    }
  }

  mods <- x[[".__enclos_env__"]][["private"]][["modules_"]]
  if (!is.null(mods)) {
    o <- mods[[y]]
    if (!is.null(o)) {
      return(o)
    }
  }

  find_method <- x[[".__enclos_env__"]][["private"]][["find_method"]]
  if (!is.null(find_method)) {
    o <- find_method(y)
    if (!is.null(o)) {
      return(o)
    }
  }

  NextMethod("[[", x)
}

#' @export
`[[<-.nn_Module` <- function(x, name, value) {
  if (inherits(value, "nn_parameter")) {
    x$register_parameter(name, value)
  } else if (inherits(value, "nn_buffer")) {
    x$register_buffer(name, value, attr(value, "persistent"))
  } else if (is_nn_module(value)) {
    x$add_module(name, value)
  } else {
    NextMethod("$<-", x)
  }

  invisible(x)
}

#' @export
`$<-.nn_Module` <- function(x, name, value) {
  x[[name]] <- value
  invisible(x)
}

#' @export
`$<-.nn_module` <- function(x, name, value) {
  attr(x, "module")[[name]] <- value
  invisible(x)
}

#' @export
`[[<-.nn_module` <- `$<-.nn_module`

#' @export
names.nn_module <- function(x, ...) {
  x <- attr(x, "module")
  NextMethod("names", x)
}

#' @export
print.nn_module <- function(x, ...) {
  x <- attr(x, "module")
  print(x)
}

#' A sequential container
#'
#' A sequential container.
#' Modules will be added to it in the order they are passed in the constructor.
#' See examples.
#'
#' @param ... sequence of modules to be added
#'
#' @examples
#'
#' model <- nn_sequential(
#'   nn_conv2d(1, 20, 5),
#'   nn_relu(),
#'   nn_conv2d(20, 64, 5),
#'   nn_relu()
#' )
#' input <- torch_randn(32, 1, 28, 28)
#' output <- model(input)
#' @export
nn_sequential <- nn_module(
  classname = "nn_sequential",
  initialize = function(...) {
    modules <- rlang::list2(...)
    for (i in seq_along(modules)) {
      self$add_module(name = i - 1, module = modules[[i]])
    }
  },
  forward = function(input) {
    for (module in private$modules_) {
      input <- module(input)
    }
    input
  }
)

#' @export
length.nn_sequential <- function(x) {
  length(x$.__enclos_env__$private$modules_)
}

#' @export
`[[.nn_sequential` <- function(x, y) {
  if (rlang::is_scalar_integerish(y)) {
    x$.__enclos_env__$private$modules_[[y]]
  } else {
    NextMethod("[[")
  }
}

#' @export
`[.nn_sequential` <- function(x, y) {
  nn_sequential(!!!lapply(y, function(i) x[[i]]))
}

#' Prune top layer(s) of a network
#'
#' Prune `head_size` last layers of a nn_module in order to
#'  replace them by your own head, or in order to use the pruned module
#'  as a sequential embedding module.
#' @param x nn_network to prune
#' @param head_size number of nn_layers to prune
#'
#' @return a nn_sequential network with the top nn_layer removed
#' @export
#'
#' @examples
#' if (torch_is_installed()) {
#' x <- nn_sequential(
#'   nn_relu(),
#'   nn_tanh(),
#'   nn_relu6(),
#'   nn_relu(),
#'   nn_linear(2,10),
#'   nn_batch_norm1d(10),
#'   nn_tanh(),
#'   nn_linear(10,3)
#' )
#' prune <- nn_prune_head(x, 3)
#' prune
#' }
nn_prune_head <- function(x, head_size) {
  UseMethod("nn_prune_head")
}

#' @export
nn_prune_head.nn_sequential <- function(x, head_size=1L ) {
  nn_sequential(!!!x$children[1:(length(x)-head_size)])
}

#' @export
nn_prune_head.nn_module <- nn_module(
    classname = "nn_sequential",
    initialize = function(x, head_size=1L) {
      modules <- rlang::list2(!!!x$children[1:(length(x$children)-head_size)])
      mod_names <- names(modules)
      for (i in seq_along(modules)) {
        self$add_module(name = mod_names[i], module = modules[[i]])
      }
    },
    forward = function(...) {
      first_module <- TRUE
      for (module in private$modules_) {
        if (first_module) {
          input <- module(...)
        } else {
          input <- module(input)
        }
        first_module <- FALSE
      }
      input
    }
  )

#' Holds submodules in a list.
#'
#' [nn_module_list] can be indexed like a regular R list, but
#' modules it contains are properly registered, and will be visible by all
#' `nn_module` methods.
#'
#' @param modules a list of modules to add
#' @seealso [nn_module_dict()]
#' @examples
#'
#' my_module <- nn_module(
#'   initialize = function() {
#'     self$linears <- nn_module_list(lapply(1:10, function(x) nn_linear(10, 10)))
#'   },
#'   forward = function(x) {
#'     for (i in 1:length(self$linears)) {
#'       x <- self$linears[[i]](x)
#'     }
#'     x
#'   }
#' )
#' @export
nn_module_list <- nn_module(
  "nn_module_list",
  initialize = function(modules = list()) {
    for (i in seq_along(modules)) {
      self$add_module(i - 1, modules[[i]])
    }
  },
  insert = function(index, module) {
    modules <- append(private$modules_, list(module), after = index - 1)
    private$modules_ <- list()
    for (i in seq_along(modules)) {
      self$add_module(i - 1, modules[[i]])
    }
  },
  append = function(module) {
    i <- length(private$modules_)
    self$add_module(i, module)
  },
  extend = function(modules) {
    for (j in seq_along(modules)) {
      self$append(modules[[j]])
    }
  }
)

#' Container that allows named values
#'
#' @param dict A named list of submodules that will be saved in that module.
#' @examples
#' nn_module <- nn_module(
#'   initialize = function() {
#'     self$dict <- nn_module_dict(list(
#'       l1 = nn_linear(10, 20),
#'       l2 = nn_linear(20, 10)
#'     ))
#'   },
#'   forward = function(x) {
#'     x <- self$dict$l1(x)
#'     self$dict$l2(x)
#'   }
#' )
#' @seealso [nn_module_list()]
#' @export
nn_module_dict <- nn_module(
  initialize = function(dict) {
    if (!rlang::is_named(dict)) cli_abort("All elements in {.arg dict} must be named.")
    for(nm in names(dict)) {
      self[[nm]] <- dict[[nm]]
    }
  },
  forward = function(...) {
    cli_abort("{.fn nn_module_dict} has {.fn forward} implementation.")
  }
)

#' @export
`[[.nn_module_list` <- function(x, y) {
  if (rlang::is_scalar_integerish(y)) {
    x$.__enclos_env__$private$modules_[[y]]
  } else {
    NextMethod("[[")
  }
}

#' @export
as.list.nn_module_list <- function(x, ...) {
  x$.__enclos_env__$private$modules_
}

#' @export
length.nn_module_list <- function(x, ...) {
  length(x$.__enclos_env__$private$modules_)
}

comma <- function(x) {
  format(x, nsmall = 0, big.mark = ",", scientific = FALSE)
}

print_nn_module <- function(self, private) {
  cli::cat_line(
    "An `nn_module` containing ",
    comma(get_parameter_count(self)),
    " parameters."
  )

  if (length(private$modules_) > 0) {
    cli::cat_line()
    cli::cat_rule("Modules")
    sapply(names(private$modules_), function(x) {
      cli_module_item(x, private$modules_[[x]])
    })
  }

  if (length(private$parameters_) > 0) {
    cli::cat_line()
    cli::cat_rule("Parameters")
    sapply(names(private$parameters_), function(x) {
      cli_tensor_item(x, private$parameters_[[x]])
    })
  }

  if (length(private$buffers_) > 0) {
    cli::cat_line()
    cli::cat_rule("Buffers")
    sapply(names(private$buffers_), function(x) {
      cli_tensor_item(x, private$buffers_[[x]])
    })
  }
}

cli_module_item <- function(name, module) {
  cli::cat_bullet(paste0(
    name,
    ": <", class(module)[1], "> #",
    comma(get_parameter_count(module)),
    " parameters"
  ))
}

cli_tensor_item <- function(name, tensor) {
  cli::cat_bullet(paste0(
    name,
    ": ",
    make_str_torch_tensor(tensor)
  ))
}

get_parameter_count <- function(self) {
  if (length(self$parameters) == 0) {
    return(0)
  }

  pars <- sapply(self$parameters, function(x) prod(x$shape))
  sum(pars)
}

keepvars_or_detach <- function(p, keepvars) {
  if (!keepvars) {
    out <- p$detach()
    out$requires_grad_(p$requires_grad)
    out
  } else {
    p
  }
}
collect_state_dict <- function(instance, state_dict) {
  # the parameters and buffers of child modules are retrieved below
  new_objs <- c(instance$named_parameters(recursive = FALSE), instance$named_buffers(recursive = FALSE))
  if (length(new_objs)) {
    names(new_objs) <- map_chr(new_objs, xptr_address)
    state_dict <- append(state_dict, new_objs)
  }
  # also need to append a clone of the modules to this list.
  # child modules can be duplicated - and have the same name
  # note that we store both the modules, as well as their parameters and buffers in the state_dict
  children <- instance$children
  if (!length(children)) {
    return(state_dict)
  }
  for (child in children) {
    state_dict <- collect_state_dict(child, state_dict)
  }
  state_dict <- append(state_dict, rlang::set_names(children, map_chr(children, rlang::obj_address)))
  return(state_dict)
}

#' @title Clone a torch module.
#'
#' @description
#' Clones a module.
#'
#' @param module ([`nn_module`])\cr
#'   The module to clone
#' @param deep (`logical(1)`)\cr
#'   Whether to create a deep clone.
#' @param ... (any)\cr
#'   Additional parameters, currently unused.
#' @param replace_values (`logical(1)`)\cr
#'   Whether to replace parameters and buffers with the cloned values.
#'
#' @export
#' @examples
#' clone_module(nn_linear(1, 1), deep = TRUE)
#' # is the same as
#' nn_linear(1, 1)$clone(deep = TRUE)
clone_module <- function(module, deep = FALSE, ..., replace_values = TRUE) {
  private <- module$.__enclos_env__$private
  if (deep && replace_values) {
    # the state_dict contains all the objects that need to be cloned and for which we also ensure that objects
    # that were previously equal by reference are still equal
    # To achieve this, the names of the state dict are the (external pointer) addresses of the objects
    # BEFORE cloning
    state_dict <- collect_state_dict(module, list())
    # each unique value must only be cloned once
    state_dict <- state_dict[!duplicated(names(state_dict))]
    state_dict <- map(state_dict, function(x) {
      if (inherits(x, "nn_module")) {
        # the values are replaced below, when calling .replace_values_from_table
        # this will fail when different submodules contain the same non-torch object by reference (e.g. R6 class), but
        # this needs a solution in R6 and not here
        x$clone(deep = deep, replace_values = FALSE)
      } else { # torch_tensor
        # without the detaching, the clone method adds a CloneBackward node which is undessireable when cloning
        # modules, as the cloned module should be independent from the clonee
        out <- x$detach()$clone()
        # we need this, because of https://github.com/mlverse/torch/issues/1136
        attributes(out) <- attributes(x)
        # because of the detach() above, we now need to reset the requires_grad field
        out$requires_grad_(x$requires_grad)
        out
      }
    })
  }
  cloned_instance <- private$.__clone_r6__(deep = deep)

  if (deep && replace_values) {
    cloned_instance$.replace_values_from_table(state_dict)
    cloned_private <- cloned_instance$.__enclos_env__$private
    if (!is.null(cloned_private$finalize_deep_clone)) {
      cloned_private$finalize_deep_clone()
    }
  }

  create_nn_module_callable(cloned_instance)
}
