nn_Module <- R6::R6Class(
  classname = "nn_Module",
  lock_objects = FALSE,
  public = list(
    training = TRUE,
    
    forward = function(...) {
      not_implemented_error("Forward methood is not implemented")
    },
    
    add_module = function(name, module) {
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
        private$non_persistent_buffers <- unique(c(
          private$non_persistent_buffers_,
          name
        ))
      }
      
    }
  ),
  private = list(
    parameters_ = list(),
    buffers_ = list(),
    modules_ = list(),
    non_persistent_buffers_ = character()
  ),
  active = list(
    parameters = function() {
      pars <- lapply(private$modules_, function(x) x$parameters)
      pars <- append(pars, private$parameters_)
      unlist(pars, recursive = TRUE, use.names = TRUE)
    }
  )
)

nn_parameter <- function(x, requires_grad = TRUE) {
  if (!is_torch_tensor(x))
    stop("`x` must be a tensor.")
  x$requires_grad_(requires_grad)
  class(x) <- c(class(x), "nn_parameter")
  x
}

is_nn_parameter <- function(x) {
  inherits(x, "nn_parameter")
}

nn_buffer <- function(x, persistent = TRUE) {
  class(x) <- c(class(x), "nn_buffer")
  attr(x, "persistent") <- persistent
  x
}

is_nn_buffer <- function(x) {
  inherits(x, "nn_buffer")
}

is_nn_module <- function(x) {
  inherits(x, "nn_module")
}

nn_module <- function(classname = NULL, inherit = nn_Module, ...) {
  
  if (inherits(inherit, "nn_module"))
    inherit <- attr(inherit, "module")
    
  Module <- R6::R6Class(
    classname = classname,
    inherit = inherit,
    lock_objects = FALSE,
    public = list(
      ...
    )
  )
  fun <- rlang::new_function(
    args = rlang::fn_fmls(Module$new), 
    body = rlang::expr({
      instance <- Module$new(!!!rlang::fn_fmls_syms(Module$new))
      f <- instance$forward
      attr(f, "class") <- "nn_module"
      attr(f, "module") <- instance
      f
    })
  )
  attr(fun, "class") <- "nn_module"
  attr(fun, "module") <- Module
  fun
}

#' @export
`$.nn_module` <- function(x, y) {
  module <- attr(x, "module")
  do.call("$", args = list(module, y))
}

#' @export
`$.nn_Module` <- function(x, y) {
  
  if (!is.null(x[[".__enclos_env__"]][["private"]][["parameters_"]])) {
    pars <- x[[".__enclos_env__"]][["private"]][["parameters_"]]
    if (y %in% names(pars))
      return(pars[[y]])
  }
  
  if (!is.null(x[[".__enclos_env__"]][["private"]][["buffers_"]])) {
    bufs <- x[[".__enclos_env__"]][["private"]][["buffers_"]]
    if (y %in% names(bufs))
      return(bufs[[y]])
  }
  
  if (!is.null(x[[".__enclos_env__"]][["private"]][["modules_"]])) {
    mods <- x[[".__enclos_env__"]][["private"]][["modules_"]]
    if (y %in% names(mods))
      return(mods[[y]])
  }
  
  NextMethod("$", x)
}


#' @export
`$<-.nn_Module` <- function(x, name, value) {
  
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
names.nn_module <- function(x, ...) {
  x <- attr(x, "module")
  NextMethod("names", x)
}

#' @export
print.nn_module <- function(x, ...) {
  x <- attr(x, "module")
  print(x)
}




