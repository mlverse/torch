as_1_based_dim <- function(x) {
  x <- as.integer(x)

  if (any(x == 0)) {
    value_error("Dimension is 1-based, but found 0.")
  }

  ifelse(x > 0, x - 1, x)
}

as_1_based_tensor_list <- function(x) {
  tensors <- lapply(tensors, as_1_based_tensor)
}

as_1_based_tensor <- function(x) {
  with_no_grad({
    if (!any(x$shape == 0)) {
      e <- torch_min(torch_abs(x))$to(dtype = torch_int())
      if (e$item() == 0) {
        runtime_error("Indices/Index start at 1 and got a 0.")
      }
    }

    out <- x - (x > 0)$to(dtype = x$dtype)
  })
  out
}

clean_chars <- c("'", "\"", "%", "#", ":", ">", "<", ",", " ", "*", "&")

clean_names <- function(x) {
  cpp_clean_names(x, clean_chars)
}

make_cpp_function_name <- function(method_name, arg_types, type) {
  cpp_make_function_name(method_name, names(arg_types), arg_types, type)
}

do_call <- function(fun, args) {
  args_needed <- names(formals(fun))
  args <- args[args_needed]
  do.call(fun, args)
}

call_c_function <- function(fun_name, args, expected_types, nd_args, return_types, fun_type) {
  fun_name <- create_fn_name(fun_name, fun_type, nd_args, args, expected_types)
  f <- getNamespace("torch")[[fun_name]]

  if (is.null(f)) {
    value_error("{fun_name} does not exist")
  }

  out <- do_call(f, args)
  out
}
