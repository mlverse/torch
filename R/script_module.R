ScriptModule <- R7Class(
  "torch_script_module",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      ptr
    },
    print = function() {
      "<script_module>"
    },
    train = function(mode = TRUE) {
      cpp_jit_script_module_train(self, mode)
      invisible(self)
    },
    register_parameter = function(name, param) {
      cpp_jit_script_module_register_parameter(self, name, param, FALSE)
      invisible(self)
    },
    register_buffer = function(name, tensor, persistent = TRUE) {
      if (!persistent)
        runtime_error("ScriptModule does not support non persistent buffers.")
      cpp_jit_script_module_register_buffer(self, name, tensor)
      invisible(self)
    },
    to = function(device, non_blocking = FALSE){
      cpp_jit_script_module_to(device, non_blocking)
      invisible(self)
    }
  ),
  active = list(
    parameters = function() {
      cpp_jit_script_module_parameters(self, TRUE)
    },
    is_training = function() {
      cpp_jit_script_module_is_training(self)
    },
    buffers = function() {
      cpp_jit_script_module_buffers(self, TRUE)
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
