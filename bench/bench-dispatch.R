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

cat("=== torch_add (multi-overload, uses generic dispatch) ===\n")
bench::mark(
  dispatch = torch_add(a, b),
  direct = resolved_fn(a, b, 1L),
  min_iterations = 10000,
  check = FALSE
) |> print()

cat("\n=== torch_matmul (single-overload, uses direct call) ===\n")
resolved_matmul <- torch:::cpp_torch_namespace_matmul_self_Tensor_other_Tensor
bench::mark(
  dispatch = torch_matmul(a, b),
  direct = resolved_matmul(a, b),
  min_iterations = 10000,
  check = FALSE
) |> print()
