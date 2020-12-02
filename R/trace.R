
jit_trace <- function(func, example_inputs, ...) {
  tr_fn <- make_traceable_fn(func)
  ex_inp <- torch_stack(example_inputs)
  ptr <- cpp_trace_function(tr_fn, ex_inp$ptr, .compilation_unit)
  new_script_function(ptr)
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
    # inputs to the traced function must be a stack
    inputs <- rlang::list2(...)
    inputs <- torch_stack(inputs)
    
    # calling the traced function always returns a stack
    # with a single element.
    out <- cpp_call_traced_fn(ptr, inputs$ptr)
    
    # post processs the output
    out <- Stack$new(ptr = out)$to_r()
    out[[1]] # always return a single thing!
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
    r_inputs <- Stack$new(ptr = inputs)$to_r()
    r_out <- do.call(fn, r_inputs)
    s_out <- Stack$new()
    s_out$push_back(r_out)
    s_out$ptr
  }
}

