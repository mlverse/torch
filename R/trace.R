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
#'   Can also be a [nn_module()], in such case [jit_trace_module()] is used to trace
#'   that module.
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
#' @returns An `script_function` if `func` is a function and `script_module` if
#'   `func` is a `nn_module()`.
#'
#' @examples
#' fn <- function(x) {
#'   torch_relu(x)
#' }
#' input <- torch_tensor(c(-1, 0, 1))
#' tr_fn <- jit_trace(fn, input)
#' tr_fn(input)
#' @export
jit_trace <- function(func, ..., strict = TRUE) {
  tr_fn <- make_traceable_fn(func)
  ellipsis::check_dots_unnamed() # we do not support named arguments

  if (inherits(func, "nn_module")) {
    if (inherits(func, "nn_module_generator")) {
      value_error("You must initialize the nn_module before tracing.")
    }

    args <- list(
      mod = func,
      forward = rlang::list2(...),
      strict = strict
    )
    return(do.call(jit_trace_module, args))
  }

  if (!rlang::is_closure(func)) {
    value_error("jit_trace needs a function or nn_module.")
  }

  ptr <- cpp_trace_function(tr_fn, list(...), .compilation_unit, strict, name = "name")
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
#' @export
jit_save <- function(obj, path, ...) {
  path <- normalizePath(path, mustWork = FALSE)

  if (inherits(obj, "script_function")) {
    obj$save(path)
  } else if (inherits(obj, "script_module")) {
    obj$..ptr..()$save(path)
  } else {
    value_error("Only `script_function` or `script_module` can be saved with `jit_save`.")
  }

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
    },
    save_for_mobile = function(filename) {
      path <- normalizePath(filename, mustWork = FALSE)
      cpp_save_traced_fn_for_mobile(self$ptr, path)
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


make_script_module_name <- function(x) {
  paste0(class(x)[1], "_", paste(sample(letters, 24, replace = TRUE), collapse = ""))
}

create_script_module <- function(mod) {
  if (inherits(mod, "script_module")) {
    return(mod)
  }

  module <- cpp_jit_script_module_new(.compilation_unit, make_script_module_name(mod))

  iwalk(mod$named_parameters(recursive = FALSE), function(par, name) {
    module$register_parameter(name, par)
  })

  iwalk(mod$named_buffers(recursive = FALSE), function(buf, name) {
    module$register_buffer(name, buf)
  })

  iwalk(mod$children, function(child, name) {
    module$register_module(name, create_script_module(child))
  })


  # Let's not keep the constants in the module right now as it might cause more
  # problems than benefits. In pytorch they are only added if their name is in
  # `__constants__` and we are using `torch.jit.script`, not `torch.jit.trace`.

  # constants <- names(mod)[!names(mod) %in% module_ignored_names]
  # walk(constants, function(name) {
  #   if (rlang::is_closure(mod[[name]])) return()
  #   # TODO catch invalid types and raise a warning listing their names.
  #   module$add_constant(name, mod[[name]])
  # })

  module
}

#' Trace a module
#'
#' Trace a module and return an executable ScriptModule that will be optimized
#' using just-in-time compilation. When a module is passed to [jit_trace()], only
#' the forward method is run and traced. With [jit_trace_module()], you can specify
#' a named list of method names to example inputs to trace (see the inputs)
#' argument below.
#'
#' See [jit_trace] for more information on tracing.
#'
#' @param mod A torch `nn_module()`  containing methods whose names are specified
#'   in inputs. The given methods will be compiled as a part of a single ScriptModule.
#' @param ... A named list containing sample inputs indexed by method names
#'   in mod. The inputs will be passed to methods whose names correspond to inputs
#'   keys while tracing. `list('forward'=example_forward_input, 'method2'=example_method2_input)`.
#'
#' @inheritParams jit_trace
#'
#' @examples
#' linear <- nn_linear(10, 1)
#' tr_linear <- jit_trace_module(linear, forward = list(torch_randn(10, 10)))
#'
#' x <- torch_randn(10, 10)
#' torch_allclose(linear(x), tr_linear(x))
#' @export
jit_trace_module <- function(mod, ..., strict = TRUE) {
  inputs <- rlang::list2(...)

  if (!inherits(mod, "nn_module")) {
    value_error("`mod` must be a `nn_module()`.")
  }

  if (!rlang::is_named(inputs)) {
    value_error("Arguments passed trough `...` must be named.")
  }

  module <- create_script_module(mod)

  for (name in names(inputs)) {
    if (!rlang::is_closure(mod[[name]])) {
      value_error("Method '{name}' does not exist in `mod` and therefore can't be traced.")
    }

    inp <- inputs[[name]]
    if (!is.list(inp)) {
      inp <- list(inp)
    }

    tr_fn <- make_traceable_fn(mod[[name]])
    ptr <- cpp_trace_function(
      fn = tr_fn,
      inputs = inp,
      compilation_unit = .compilation_unit,
      strict = strict,
      module = module$..ptr..(),
      name = name,
      should_mangle = TRUE,
      manage_memory = FALSE
    )
    cpp_jit_script_module_add_method(module$..ptr..(), ptr)
  }

  module
}

#' Saves a `script_function` or `script_module` in bytecode form,
#' to be loaded on a mobile device
#'
#' @param obj An `script_function` or `script_module` to save
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
#' jit_save_for_mobile(tr_fn, tmp)
#' @export
jit_save_for_mobile <- function(obj, path, ...) {
  path <- normalizePath(path, mustWork = FALSE)

  if (inherits(obj, "script_function")) {
    obj$save_for_mobile(path)
  } else if (inherits(obj, "script_module")) {
    obj$..ptr..()$save_for_mobile(path)
  } else {
    value_error("Only `script_function` or `script_module` can be saved with `jit_save`.")
  }

  invisible(obj)
}
