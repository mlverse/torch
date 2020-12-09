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
#' @param example_inputs example inputs that will be passed to the function while 
#'   tracing. The resulting trace can be run with inputs of different types and 
#'   shapes assuming the traced operations support those types and shapes. 
#'   `example_inputs` may also be a single Tensor in which case it is automatically 
#'   wrapped in a list.
#' @param ... currently unused.
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
jit_trace <- function(func, example_inputs, ...) {
  tr_fn <- make_traceable_fn(func)
  ex_inp <- torch_jit_stack(example_inputs)
  ptr <- cpp_trace_function(tr_fn, ex_inp$ptr, .compilation_unit)
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
  ptr <- cpp_jit_load(path)
  new_script_module(ptr)
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

ScriptModule <- R6::R6Class(
  "ScriptModule",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    }
  )
)

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

convert_inputs_to_jit_stack <- function(...) {
  # inputs to the traced function must be a stack
  inputs <- torch_jit_stack(...)
  inputs
}

convert_outputs_to_r <- function(out) {
  # post processs the output
  out <- Stack$new(ptr = out)$to_r()
  out[[1]] # always return a single thing!
}

new_script_function <- function(ptr) {
  f <- function(...) {
    inputs <- convert_inputs_to_jit_stack(...)
    # calling the traced function always returns a stack
    # with a single element.
    out <- cpp_call_traced_fn(ptr, inputs$ptr)
    convert_outputs_to_r(out)
  }
  class(f) <- "script_function"
  attr(f, "ScriptFunction") <- ScriptFunction$new(ptr = ptr)
  f
}

new_script_module <- function(ptr) {
  f <- function(...) {
    inputs <- convert_inputs_to_jit_stack(...)
    # calling the traced function always returns a stack
    # with a single element.
    out <- cpp_call_jit_script(ptr, inputs$ptr)
    convert_outputs_to_r(out)
  }
  class(f) <- "script_module"
  attr(f, "ScriptModule") <- ScriptModule$new(ptr = ptr)
  f
}

#' @export
print.script_function <- function(x, ...) {
  cat("<script_function>\n")
}

#' @export
print.jit_module <- function(x, ...) {
  cat("script_module>\n")
}

#' @export
`$.script_function` <- function(x, name) {
  attr(x, "ScriptFunction")[[name]]
}

# Wraps a function in an R function that takes the inputs as
# a torch::jit::Stack ptr and returns a torch::jit::Stack ptr.
make_traceable_fn <- function(fn) {
  function(inputs) {
    r_inputs <- Stack$new(ptr = inputs)$to_r()
    r_out <- do.call(fn, r_inputs)
    s_out <- Stack$new()
    s_out$push_back(r_out)
    s_out$ptr
  }
}

