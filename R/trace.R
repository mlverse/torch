#' Trace a function and return an executable `script_function`.
#' 
#' Using `jit_trace`, you can turn an existing R function into a TorchScript 
#' `script_function`. You must provide example inputs, and we run the function, 
#' recording the operations performed on all the tensors.
#' 
#' The resulting recording of a standalone function produces a `script_function`.
#' In the future we will also support tracing `nn_modules`.
#' 
#' @section Warning:
#' 
#' Tracing only correctly records functions and modules which are not data dependent 
#' (e.g., do not have conditionals on data in tensors) and do not have any untracked 
#' external dependencies (e.g., perform input/output or access global variables). 
#' Tracing only records operations done when the given function is run on the given 
#' tensors. Therefore, the returned `script_function` will always run the same traced 
#' graph on any input. This has some important implications when your module is 
#' expected to run different sets of operations, depending on the input and/or the
#' module state. For example,
#' 
#' * Tracing will not record any control-flow like if-statements or loops. When 
#'   this control-flow is constant across your module, this is fine and it often 
#'   inlines the control-flow decisions. But sometimes the control-flow is actually 
#'   part of the model itself. For instance, a recurrent network is a loop over 
#'   the (possibly dynamic) length of an input sequence.
#' * In the returned `script_function`, operations that have different behaviors 
#'   in training and eval modes will always behave as if it is in the mode it was 
#'   in during tracing, no matter which mode the `script_function` is in.
#'
#' In cases like these, tracing would not be appropriate and scripting is a better 
#' choice. If you trace such models, you may silently get incorrect results on 
#' subsequent invocations of the model. The tracer will try to emit warnings when 
#' doing something that may cause an incorrect trace to be produced.
#' 
#' @note Scripting is not yet supported in R.
#' 
#' @param func An R function that will be run with `example_inputs`. func arguments 
#'   and return values must be tensors or (possibly nested) lists that contain tensors.
#' @param ... example inputs that will be passed to the function while 
#'   tracing. The resulting trace can be run with inputs of different types and 
#'   shapes assuming the traced operations support those types and shapes. 
#'   `example_inputs` may also be a single Tensor in which case it is automatically 
#'   wrapped in a list. Note that `...` **can not** be named, and the order is 
#'   respected.
#' @param strict run the tracer in a strict mode or not (default: `TRUE`). Only 
#'   turn this off when you want the tracer to record your mutable container types 
#'   (currently list/dict) and you are sure that the container you are using in 
#'   your problem is a constant structure and does not get used as control flow 
#'   (`if`, `for`) conditions.
#' 
#' @returns An `script_function`
#' 
#' @examples
#' fn <- function(x) {
#'  torch_relu(x)
#' }
#' input <- torch_tensor(c(-1, 0, 1))
#' tr_fn <- jit_trace(fn, input)
#' tr_fn(input)
#'
#' @export
jit_trace <- function(func, ..., strict = TRUE) {
  tr_fn <- make_traceable_fn(func)
  ellipsis::check_dots_unnamed() # we do not support named arguments
  ptr <- cpp_trace_function(tr_fn, list(...), .compilation_unit, strict)
  new_script_function(ptr)
}

#' Loads a `script_function` or `script_module` previously saved with `jit_save`
#' 
#' @param path a path to a `script_function` or `script_module` serialized with
#'   [jit_save()].
#' @param ... currently unused.
#' 
#' @export
jit_load <- function(path, ...) {
  path <- normalizePath(path, mustWork = TRUE)
  cpp_jit_load(path)
}

#' Saves a `script_function` to a path
#' 
#' @param obj An `script_function` to save
#' @param path The path to save the serialized function.
#' @param ... currently unused
#' 
#' @examples
#' fn <- function(x) {
#'   torch_relu(x)
#' }
#' 
#' input <- torch_tensor(c(-1, 0, 1))
#' tr_fn <- jit_trace(fn, input)
#' 
#' tmp <- tempfile("tst", fileext = "pt")
#' jit_save(tr_fn, tmp)
#' 
#' @export
jit_save <- function(obj, path, ...) {
  path <- normalizePath(path, mustWork = FALSE)
  obj$save(path)
  invisible(obj)
}

ScriptFunction <- R6::R6Class(
  "ScriptFunction",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    save = function(filename) {
      path <- normalizePath(filename, mustWork = FALSE)
      cpp_save_traced_fn(self$ptr, path)
      invisible(self)
    }
  ),
  active = list(
    graph = function() {
      GraphFunction$new(ptr = self$ptr)
    }
  )
)

GraphFunction <- R6::R6Class(
  classname = "graph_function",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    print = function() {
      cat(cpp_traced_fn_graph_print(self$ptr))
    }
  )
)

new_script_function <- function(ptr) {
  f <- function(...) {
    inputs <- list(...)
    out <- cpp_call_traced_fn(ptr, inputs)
    # calling the traced function always returns a stack
    # with a single element.
    out[[1]]
  }
  class(f) <- "script_function"
  attr(f, "ScriptFunction") <- ScriptFunction$new(ptr = ptr)
  f
}

#' @export
print.script_function <- function(x, ...) {
  cat("<script_function>\n")
}

#' @export
`$.script_function` <- function(x, name) {
  attr(x, "ScriptFunction")[[name]]
}

# Wraps a function in an R function that takes the inputs as
# a torch::jit::Stack ptr and returns a torch::jit::Stack ptr.
make_traceable_fn <- function(fn) {
  function(inputs) {
    out <- do.call(fn, inputs)
    list(out)
  }
}


module_ignored_names <- c(
  ".__enclos_env__", "children", 
  "modules", "buffers", "parameters", ".classes", "training", "clone", 
  "forward", "reset_parameters", "initialize", "named_buffers", 
  "named_parameters", "apply", "zero_grad", "load_state_dict", 
  ".load_from_state_dict", "state_dict", ".save_to_state_dict", 
  "print", "to", "cpu", "cuda", ".apply", "eval", "train", "register_buffer", 
  "register_parameter", "register_module", "add_module"
)


create_script_module <- function(mod) {
  
  module <- cpp_jit_script_module_new(.compilation_unit, digest::digest(runif(1)))
  
  iwalk(mod$named_parameters(recursive = FALSE), function(par, name) {
    module$register_parameter(name, par)
  })
  
  iwalk(mod$named_buffers(recursive = FALSE), function(buf, name) {
    module$register_buffer(name, buf)
  })
  
  iwalk(mod$children, function(child, name) {
    module$register_module(name, create_script_module(child))
  })
  
  constants <- names(mod)[!names(mod) %in% module_ignored_names]
  walk(constants, function(name) {
    if (rlang::is_closure(mod[[name]])) return()
    # TODO catch invalid types and raise a warning listing their names.
    module$add_constant(name, mod[[name]])  
  })
  
  module
}

jit_trace_module <- function(mod, inputs) {
  module <- create_script_module(mod)
}

