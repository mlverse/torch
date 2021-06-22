ScriptModule <- R7Class(
  "torch_script_module",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      ptr
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
      cpp_jit_script_module_to(self, device, non_blocking)
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
    },
    modules = function() {
      cpp_jit_script_module_children(self)
    }
  )
)

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
            for(name in names(value)) {
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
            for(name in names(value)) {
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
    }
  )
)

new_script_module <- function(ptr) {
  f <- function(...) {
    inputs <- convert_inputs_to_jit_stack(...)
    # calling the traced function always returns a stack
    # with a single element.
    out <- cpp_call_jit_script(ptr, inputs$ptr)
    convert_outputs_to_r(out)
  }
  class(f) <- c("script_module", "nn_module")
  attr(f, "module") <- nn_ScriptModule$new(ptr = ptr)
  f
}
