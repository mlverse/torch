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
