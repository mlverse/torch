ark_register_function <- ".ark.register_method"

positron_is_enabled_to_register <- function() {
  exists(ark_register_function, mode = "function")
}

register_ark_method <- function(generic, class, method) {  
  register_call <- call(ark_register_function, quote(generic), quote(class), quote(method))
  eval(register_call, envir = environment())
}

register_positron_methods <- function() {
  if (!positron_is_enabled_to_register()) {
    return()
  }

  tryCatch({
    register_positron_methods_impl()
  }, error = function(err) {
    cli::cli_warn("Failed to register Positron methods.", parent = err)
  })
}

register_positron_methods_impl <- function() {
  register_ark_method("ark_positron_variable_display_value", "torch_tensor", function(x, ...) {
    make_str_torch_tensor(x)
  })

  register_ark_method("ark_positron_variable_display_value", "nn_module", function(x, ...) {
    paste0(
      "nn_module (",
      scales::comma(get_parameter_count(attr(x, "module"))),
      " parameters)"
    )
  })

  register_ark_method("ark_positron_variable_get_children", "nn_module", function(x, ...) {
    x$parameters
  })
  
  register_ark_method("ark_positron_variable_has_children", "nn_module", function(x, ...) {
    TRUE
  })
}


