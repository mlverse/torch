resolve_size <- function(...) {
  size <- rlang::list2(...)

  if (!is.null(size[["size"]])) {
    if (!length(size) == 1) {
      stop("You should specify a single size argument.")
    }

    size <- size[["size"]]
  } else if (length(size[[1]]) > 1) {
    if (!length(size) == 1) {
      stop("You should specify a single size argument.")
    }

    size <- size[[1]]
  }

  size
}

#' @rdname torch_ones
torch_ones <- function(..., names = NULL, dtype = NULL, layout = torch_strided(),
                       device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_ones, args)
}

#' @rdname torch_ones_like
torch_ones_like <- function(input, dtype = NULL, layout = torch_strided(),
                            device = NULL, requires_grad = FALSE,
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_ones_like, args)
}

#' @rdname torch_rand
torch_rand <- function(..., names = NULL, dtype = NULL, layout = torch_strided(),
                       device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_rand, args)
}

#' @rdname torch_rand_like
torch_rand_like <- function(input, dtype = NULL, layout = torch_strided(),
                            device = NULL, requires_grad = FALSE,
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_rand_like, args)
}

#' @rdname torch_randint
torch_randint <- function(low, high, size, generator = NULL, dtype = NULL, layout = torch_strided(),
                          device = NULL, requires_grad = FALSE,
                          memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$low <- low
  args$high <- high
  args$size <- size
  if (!is.null(generator)) args$generator <- NULL

  do.call(.torch_randint, args)
}

#' @rdname torch_randint_like
torch_randint_like <- function(input, low, high, dtype = NULL,
                               layout = torch_strided(),
                               device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$self <- input
  args$low <- low
  args$high <- high

  do.call(.torch_randint_like, args)
}

#' @rdname torch_randn
torch_randn <- function(..., names = NULL, dtype = NULL, layout = torch_strided(),
                        device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_randn, args)
}

#' @rdname torch_randn_like
torch_randn_like <- function(input, dtype = NULL, layout = torch_strided(),
                             device = NULL, requires_grad = FALSE,
                             memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_randn_like, args)
}

#' @rdname torch_randperm
torch_randperm <- function(n, dtype = torch_int64(), layout = torch_strided(),
                           device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$n <- n
  do.call(.torch_randperm, args)
}

#' @rdname torch_zeros
torch_zeros <- function(..., names = NULL, dtype = NULL, layout = torch_strided(),
                        device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_zeros, args)
}

#' @rdname torch_zeros_like
torch_zeros_like <- function(input, dtype = NULL, layout = torch_strided(),
                             device = NULL, requires_grad = FALSE,
                             memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_zeros_like, args)
}

#' @rdname torch_empty
torch_empty <- function(..., names = NULL, dtype = NULL, layout = torch_strided(),
                        device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_empty, args)
}

#' @rdname torch_empty_like
torch_empty_like <- function(input, dtype = NULL, layout = torch_strided(),
                             device = NULL, requires_grad = FALSE,
                             memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_empty_like, args)
}

#' @rdname torch_arange
torch_arange <- function(start, end, step = 1, dtype = NULL, layout = torch_strided(),
                         device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$start <- start

  eps <- if (is.null(dtype) || ((!dtype == torch_float32()) && (!dtype == torch_float64()))) {
    torch_finfo(torch_get_default_dtype())$eps
  } else {
    torch_finfo(dtype)$eps
  }
  args$end <- end + sign(step) * eps

  args$step <- step
  do.call(.torch_arange, args)
}

#' @rdname torch_range
torch_range <- function(start, end, step = 1, dtype = NULL, layout = torch_strided(),
                        device = NULL, requires_grad = FALSE) {
  warning("This function is deprecated in favor of torch_arange.")
  torch_arange(
    start = start, end = end, step = step, dtype = dtype, layout = layout,
    device = device, requires_grad = requires_grad
  )
}

#' @rdname torch_linspace
torch_linspace <- function(start, end, steps = 100, dtype = NULL, layout = torch_strided(),
                           device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$start <- start
  args$end <- end
  args$steps <- steps
  do.call(.torch_linspace, args)
}

#' @rdname torch_logspace
torch_logspace <- function(start, end, steps = 100, base = 10, dtype = NULL, layout = torch_strided(),
                           device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$start <- start
  args$end <- end
  args$steps <- steps
  args$base <- base
  do.call(.torch_logspace, args)
}

#' @rdname torch_eye
torch_eye <- function(n, m = n, dtype = NULL, layout = torch_strided(),
                      device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$n <- n
  args$m <- m
  do.call(.torch_eye, args)
}

#' @rdname torch_empty_strided
torch_empty_strided <- function(size, stride, dtype = NULL, layout = torch_strided(),
                                device = NULL, requires_grad = FALSE, pin_memory = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad,
      pinned_memory = pin_memory
    )
  )
  args$size <- size
  args$stride <- stride
  do.call(.torch_empty_strided, args)
}

#' @rdname torch_full
torch_full <- function(size, fill_value, names = NULL, dtype = NULL, layout = torch_strided(),
                       device = NULL, requires_grad = FALSE) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  if (!is.null(names)) args$names <- names
  args$size <- size
  args$fill_value <- fill_value
  do.call(.torch_full, args)
}

#' @rdname torch_full_like
torch_full_like <- function(input, fill_value, dtype = NULL, layout = torch_strided(),
                            device = NULL, requires_grad = FALSE,
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = torch_tensor_options(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  args$fill_value <- fill_value
  do.call(.torch_full_like, args)
}

#' Scalar tensor
#'
#' Creates a singleton dimension tensor.
#'
#' @param value the value you want to use
#' @inheritParams torch_ones
#'
#' @export
torch_scalar_tensor <- function(value, dtype = NULL, device = NULL, requires_grad = FALSE) {
  if (is_torch_tensor(value) && !is.null(value$shape) && sum(value$shape) > 1) {
    value_error("values must be lenght 1")
  }

  if (!is_torch_tensor(value) && length(value) > 1) {
    value_error("value must be a lenght 1 vector")
  }

  if (is_torch_tensor(value)) {
    value$squeeze()$to(device = device, dtype = dtype)$requires_grad_(requires_grad)
  } else {
    torch_tensor(value, dtype = dtype, device = device, requires_grad = requires_grad)$squeeze()
  }
}
