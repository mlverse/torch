nn_ScriptModule <- R6::R6Class(
  inherit = nn_Module,
  lock_objects = FALSE,
  public = list(
    initialize = function(ptr) {
      private$ptr <- ptr

      rm(list = "parameters_", envir = private)
      rm(list = "buffers_", envir = private)
      rm(list = "modules_", envir = private)

      makeActiveBinding(
        "parameters_",
        fun = function(value) {
          if (!missing(value)) {
            for (name in names(value)) {
              self$register_parameter(name, value[[name]])
            }
          }

          cpp_jit_script_module_parameters(private$ptr, recurse = FALSE)
        },
        env = private
      )

      makeActiveBinding(
        "buffers_",
        fun = function(value) {
          if (!missing(value)) {
            for (name in names(value)) {
              self$register_buffer(name, value[[name]])
            }
          }

          cpp_jit_script_module_buffers(private$ptr, recurse = FALSE)
        },
        env = private
      )

      makeActiveBinding(
        "modules_",
        fun = function() {
          private$ptr$modules
        },
        env = private
      )
    },
    register_parameter = function(name, param) {
      private$ptr$register_parameter(name, param)
    },
    register_buffer = function(name, tensor, persistent = TRUE) {
      private$ptr$register_buffer(name, tensor, persistent)
    },
    register_module = function(name, module) {
      private$ptr$register_module(name, module)
    },
    add_constant = function(name, value) {
      private$ptr$add_constant(name, value)
    },
    graph_for = function(...) {
      self$forward$graph_for(...)
    },
    ..ptr.. = function() {
      private$ptr
    }
  ),
  private = list(
    find_method = function(name) {
      private$ptr$find_method(name)
    }
  ),
  active = list(
    graph = function() {
      self$forward$graph
    }
  )
)

#' @export
`[[.script_module` <- function(x, y) {
  out <- attr(x, "module")$..ptr..()$find_constant(y)
  if (!is.null(out)) {
    return(out)
  }
  NextMethod()
}

#' @export
`$.script_module` <- function(x, y) {
  x[[y]]
}

new_script_module <- function(ptr) {
  f <- function(...) {
    inputs <- list(...)

    if (is.null(ptr$find_method("forward"))) {
      runtime_error("Forward is not defined. Methods from submodules of traced modules are not traced. Are you trying to call from a submodule?")
    }

    out <- cpp_call_jit_script(ptr, inputs)
    # calling the traced function always returns a stack
    # with a single element.
    out[[1]]
  }
  class(f) <- c("script_module", "nn_module")
  attr(f, "module") <- nn_ScriptModule$new(ptr = ptr)
  f
}

ScriptMethod <- R7Class(
  "torch_script_method",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      ptr
    },
    print = function() {
      cat("<script_method>\n")
    },
    graph_for = function(...) {
      # we only implement python's fallback method which calls the graph and
      # then reads the last used method.
      new_script_method(self)(...)
      str <- cpp_jit_last_executed_optimized_graph_print()
      structure(list(str = str), class = "script_method_graph")
    }
  ),
  active = list(
    graph = function() {
      str <- cpp_jit_script_method_graph_print(self)
      structure(list(str = str), class = "script_method_graph")
    }
  )
)

new_script_method <- function(ptr) {
  f <- function(...) {
    out <- cpp_jit_script_method_call(ptr, list(...))
    # calling the traced function always returns a stack
    # with a single element.
    out[[1]]
  }
  class(f) <- c("script_method")
  attr(f, "method") <- ptr
  f
}

#' @export
`[[.script_method` <- function(x, y) {
  attr(x, "method")[[y]]
}

#' @export
`$.script_method` <- function(x, y) {
  x[[y]]
}

#' @export
print.script_method <- function(x, ...) {
  cat("<script_method>\n")
}

#' @export
print.script_method_graph <- function(x, ...) {
  cat(x$str)
}
