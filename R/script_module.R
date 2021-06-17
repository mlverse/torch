ScriptModule <- R7Class(
  "torch_script_module",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      ptr
    },
    print = function() {
      "<script_module>"
    }
  ),
  active = list(
    parameters = function() {
      cpp_jit_script_module_parameters(self)
    }
  )
)

#' @export
`$.script_module` <- function(x, nm) {
  attr(x, "ScriptModule")[[nm]]
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
  attr(f, "ScriptModule") <- ptr
  f
}

#' @export
print.script_module <- function(x, ...) {
  cat("script_module>\n")
}
