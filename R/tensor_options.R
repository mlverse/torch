

torch_tensor_options <- function(dtype = NULL, layout = NULL, device = NULL,
                                 requires_grad = NULL, pinned_memory = NULL) {
  options <- list(
    dtype = if (is.character(dtype)) dtype else dtype$ptr,
    layout = layout$ptr,
    device = if (is.character(device)) torch_device(device)$ptr else device$ptr,
    requires_grad = requires_grad,
    pinned_memory = pinned_memory
  )
  options <- Filter(Negate(is.null), options)
  options
}

is_torch_tensor_options <- function(x) {
  inherits(x, "torch_tensor_options")
}

as_torch_tensor_options <- function(l) {
  do.call(torch_tensor_options, l)
}
