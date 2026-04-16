library(torch)
Sys.setenv(KMP_DUPLICATE_LIB_OK = TRUE)

a <- torch_randn(2, 2)
b <- torch_randn(2, 2)

# Pre-resolve the target function to isolate different cost components
resolved_fn <- torch:::cpp_torch_namespace_add_self_Tensor_other_Tensor

# Simulate what call_c_function does
args <- list(self = a, other = b, alpha = 1L)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")

bench::mark(
  # Full dispatch: type resolution + name building + namespace lookup + do_call
  dispatch = torch_add(a, b),
  # Just the name resolution (C++ side)
  create_fn_name = torch:::create_fn_name("add", "namespace", nd_args, args, expected_types),
  # Just the namespace lookup
  ns_lookup = getNamespace("torch")[["cpp_torch_namespace_add_self_Tensor_other_Tensor"]],
  # Just the do_call overhead (formals + do.call)
  do_call = torch:::do_call(resolved_fn, args),
  # Skip dispatch, call the Rcpp wrapper directly
  direct = resolved_fn(a, b, 1L),
  min_iterations = 10000,
  check = FALSE
)
