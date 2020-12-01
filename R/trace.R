
jit_trace <- function(func, example_inputs, ...) {
  tr_fn <- make_traceable_fn(func)
  ex_inp <- torch_stack(example_inputs)
  ptr <- cpp_trace_function(fn, input$ptr, .compilation_unit)
  new_script_function(ptr)
}

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
  f
}

#' @export
print.script_function <- function(x, ...) {
  cat("<script_function>\n")
}

# Wraps a function in an R function that takes the inputs as
# a torch::jit::Stack ptr and returns a torch::jit::Stack ptr.
make_traceable_fn <- function(fn) {
  function(inputs) {
    r_inputs <- Stack$new(ptr = inputs)$to_r()
    r_out <- do.call(fn, inputs)
    s_out <- Stack$new()
    s_out$push_back(r_out)
    s_out$ptr
  }
}

