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
  to_index_tensor(x)
}

clean_chars <- c("'", "\"", "%", "#", ":", ">", "<", ",", " ", "*", "&")

clean_names <- function(x) {
  cpp_clean_names(x, clean_chars)
}

make_cpp_function_name <- function(method_name, arg_types, type) {
  cpp_make_function_name(method_name, names(arg_types), arg_types, type)
}

# Type-check helpers used by generated inline dispatch code.
# These replicate the priority logic in cpp_arg_to_torch_type for specific
# type pairs so that codegen can emit simple if/else branches.
is_tensor_dispatch <- function(x) {
  if (inherits(x, "torch_tensor")) return(TRUE)
  if (is.null(x) || inherits(x, "torch_scalar")) return(FALSE)
  is.atomic(x) && length(x) != 1L
}

is_int64_dispatch <- function(x) {
  is.numeric(x) && length(x) == 1L
}

do_call <- function(fun, args) {
  args_needed <- names(formals(fun))
  args <- args[args_needed]
  do.call(fun, args)
}

.dispatch_cache <- new.env(parent = emptyenv())

call_c_function <- function(fun_name, args, expected_types, nd_args, return_types, fun_type) {
  fun_name <- create_fn_name(fun_name, fun_type, nd_args, args, expected_types)

  cached <- .dispatch_cache[[fun_name]]
  if (is.null(cached)) {
    f <- getNamespace("torch")[[fun_name]]
    if (is.null(f)) {
      value_error("{fun_name} does not exist")
    }
    cached <- list(f = f, args_needed = names(formals(f)))
    .dispatch_cache[[fun_name]] <- cached
  }

  do.call(cached$f, args[cached$args_needed])
}
