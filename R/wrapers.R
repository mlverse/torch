#' LU
#'
#' Computes the LU factorization of a matrix or batches of matrices A. Returns a
#' tuple containing the LU factorization and pivots of A. Pivoting is done if pivot
#' is set to True.
#'
#' @param A (Tensor) the tensor to factor of size (*, m, n)(*,m,n)
#' @param pivot (bool, optional) – controls whether pivoting is done. Default: TRUE
#' @param get_infos (bool, optional) – if set to True, returns an info IntTensor. Default: FALSE
#' @param out (tuple, optional) – optional output tuple. If get_infos is True, then the elements
#'   in the tuple are Tensor, IntTensor, and IntTensor. If get_infos is False, then the
#'   elements in the tuple are Tensor, IntTensor. Default: NULL
#'
#' @examples
#'
#' A <- torch_randn(c(2, 3, 3))
#' torch_lu(A)
#' @export
torch_lu <- function(A, pivot = TRUE, get_infos = FALSE, out = NULL) {
  # If get_infos is True, then we don't need to check for errors and vice versa
  result <- torch__lu_with_info(A, pivot, get_infos)

  if (!is.null(out)) {
    if (!is.list(out)) {
      stop("argument 'out' must be a list of Tensors.")
    }

    if (length(out) - as.integer(get_infos) != 2) {
      stop(
        "expected tuple of ", 2 + as.integer(get_infos), " elements but got ",
        length(out)
      )
    }

    for (i in seq_len(out)) {
      out[[i]] <- out[[i]]$resize_as_(result[[i]])$copy_(result[[i]])
    }

    return(out)
  }


  if (get_infos) {
    return(result)
  } else {
    return(result[1:2])
  }
}

torch_logical_not <- function(self) {
  .torch_logical_not(self)
}

#' @rdname torch_bartlett_window
torch_bartlett_window <- function(window_length, periodic = TRUE, dtype = NULL,
                                  layout = torch_strided(), device = NULL,
                                  requires_grad = FALSE) {
  opt <- torch_tensor_options(
    dtype = dtype, layout = layout, device = device,
    requires_grad = requires_grad
  )
  .torch_bartlett_window(
    window_length = window_length, periodic = periodic,
    options = opt
  )
}

#' @rdname torch_blackman_window
torch_blackman_window <- function(window_length, periodic = TRUE, dtype = NULL,
                                  layout = torch_strided(), device = NULL,
                                  requires_grad = FALSE) {
  opt <- torch_tensor_options(
    dtype = dtype, layout = layout, device = device,
    requires_grad = requires_grad
  )
  .torch_blackman_window(
    window_length = window_length, periodic = periodic,
    options = opt
  )
}

#' @rdname torch_hamming_window
torch_hamming_window <- function(window_length, periodic = TRUE, alpha = 0.54,
                                 beta = 0.46, dtype = NULL, layout = torch_strided(),
                                 device = NULL, requires_grad = FALSE) {
  opt <- torch_tensor_options(
    dtype = dtype, layout = layout, device = device,
    requires_grad = requires_grad
  )
  .torch_hamming_window(
    window_length = window_length, periodic = periodic,
    alpha = alpha, beta = beta, options = opt
  )
}

#' @rdname torch_hann_window
torch_hann_window <- function(window_length, periodic = TRUE, dtype = NULL,
                              layout = torch_strided(), device = NULL,
                              requires_grad = FALSE) {
  if (is.null(dtype)) {
    dtype <- torch_float()
  }

  opt <- torch_tensor_options(
    dtype = dtype, layout = layout, device = device,
    requires_grad = requires_grad
  )

  if (is.null(window_length)) {
    value_error("argument 'window_length' must be int, not NULL")
  }

  .torch_hann_window(
    window_length = window_length, periodic = periodic,
    options = opt
  )
}

#' @rdname torch_result_type
torch_result_type <- function(tensor1, tensor2) {
  if (is_torch_tensor(tensor1) && is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_other_Tensor_tensor_Tensor(
      tensor1$ptr,
      tensor2$ptr
    )
  } else if (is_torch_tensor(tensor1) && !is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_other_Scalar_tensor_Tensor(
      tensor1$ptr,
      torch_scalar(tensor2)$ptr
    )
  } else if (!is_torch_tensor(tensor1) && is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_scalar_Scalar_tensor_Tensor(
      torch_scalar(tensor1)$ptr,
      tensor2$ptr
    )
  } else if (!is_torch_tensor(tensor1) && !is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_scalar1_Scalar_scalar2_Scalar(
      torch_scalar(tensor1)$ptr,
      torch_scalar(tensor2)$ptr
    )
  }

  torch_dtype$new(ptr = o)
}

#' @rdname torch_sparse_coo_tensor
torch_sparse_coo_tensor <- function(indices, values, size = NULL, dtype = NULL,
                                    device = NULL, requires_grad = FALSE) {
  opt <- torch_tensor_options(
    dtype = dtype, device = device,
    requires_grad = requires_grad
  )

  if (is.null(size)) {
    .torch_sparse_coo_tensor(indices, values, options = opt)
  } else {
    .torch_sparse_coo_tensor(indices, values, size = size, options = opt)
  }
}

#' @rdname torch_stft
torch_stft <- function(input, n_fft, hop_length = NULL, win_length = NULL,
                       window = NULL, center = TRUE, pad_mode = "reflect",
                       normalized = FALSE, onesided = TRUE, return_complex = NULL) {
  if (center) {
    signal_dim <- input$dim()
    extended_shape <- c(
      rep(1, 3 - signal_dim),
      input$size()
    )
    pad <- as.integer(n_fft %/% 2)
    input <- nnf_pad(
      input = input$view(extended_shape), pad = c(pad, pad),
      mode = pad_mode
    )
    input <- input$view(utils::tail(input$shape, signal_dim))
  }

  if (is.null(return_complex)) {
    return_complex <- FALSE
  }

  .torch_stft(
    self = input, n_fft = n_fft, hop_length = hop_length,
    win_length = win_length, window = window,
    normalized = normalized, onesided = onesided,
    return_complex = return_complex
  )
}

#' @rdname torch_tensordot
torch_tensordot <- function(a, b, dims = 2) {
  if (is.list(dims)) {
    dims_a <- dims[[1]]
    dims_b <- dims[[2]]
  } else if (is_torch_tensor(dims) && dims$numel() > 1) {
    dims_a <- as_array(dims[1])
    dims_b <- as_array(dims[2])
  } else {
    if (is_torch_tensor(dims)) {
      dims <- dims$item()
    }

    if (dims < 1) {
      runtime_error("tensordot expects dims >= 1, but got {dims}")
    }

    dims_a <- seq(from = -dims, to = -1)
    dims_b <- seq(from = 1, to = dims)
  }

  .torch_tensordot(a, b, dims_a, dims_b)
}

#' @rdname torch_tril_indices
torch_tril_indices <- function(row, col, offset = 0, dtype = torch_long(),
                               device = "cpu", layout = torch_strided()) {
  opt <- torch_tensor_options(dtype = dtype, device = device, layout = layout)
  .torch_tril_indices(row, col, offset, options = opt)
}

#' @rdname torch_triu_indices
torch_triu_indices <- function(row, col, offset = 0, dtype = torch_long(),
                               device = "cpu", layout = torch_strided()) {
  opt <- torch_tensor_options(dtype = dtype, device = device, layout = layout)
  .torch_triu_indices(row, col, offset, options = opt)
}

#' @rdname torch_multilabel_margin_loss
torch_multilabel_margin_loss <- function(self, target, reduction = torch_reduction_mean()) {
  .torch_multilabel_margin_loss(self, as_1_based_tensor(target), reduction)
}

#' @rdname torch_multi_margin_loss
torch_multi_margin_loss <- function(self, target, p = 1L, margin = 1L, weight = list(),
                                    reduction = torch_reduction_mean()) {
  .torch_multi_margin_loss(
    self, as_1_based_tensor(target), p, margin, weight,
    reduction
  )
}

#' @rdname torch_topk
torch_topk <- function(self, k, dim = -1L, largest = TRUE, sorted = TRUE) {
  o <- .torch_topk(self, k, dim, largest, sorted)
  o[[2]]$add_(1L)
  o
}

#' @rdname torch_narrow
torch_narrow <- function(self, dim, start, length) {
  start <- torch_scalar_tensor(start, dtype = torch_int64())

  if (start$item() == 0) {
    value_error("start indexing starts at 1")
  }

  start <- start - 1L

  .torch_narrow(self, dim, start, length)
}

#' @rdname torch_quantize_per_tensor
torch_quantize_per_tensor <- function(self, scale, zero_point, dtype) {
  args <- list()

  if (is.list(self)) {
    args$tensors <- self
  } else {
    args$self <- self
  }

  if (is.list(scale)) {
    args$scales <- scale
  } else {
    args$scale <- scale
  }

  if (is.list(zero_point)) {
    args$zero_points <- zero_point
  } else {
    args$zero_point <- zero_point
  }

  args$dtype <- dtype

  do.call(.torch_quantize_per_tensor, args)
}

#' @rdname torch_upsample_nearest1d
torch_upsample_nearest1d <- function(input, self, output_size = NULL,
                                     scale_factors = NULL,
                                     scales = NULL) {
  args <- list(
    input = input, output_size = output_size,
    scale_factors = scale_factors, scales = scales
  )

  if (!missing(self)) {
    args$self <- self
  }

  do.call(.torch_upsample_nearest1d, args)
}

#' @rdname torch_upsample_nearest2d
torch_upsample_nearest2d <- function(input, self, output_size = NULL,
                                     scale_factors = NULL,
                                     scales_h = NULL, scales_w = NULL) {
  args <- list(
    input = input, output_size = output_size,
    scale_factors = scale_factors,
    scales_h = scales_h, scales_w = scales_w
  )

  if (!missing(self)) {
    args$self <- self
  }

  do.call(.torch_upsample_nearest2d, args)
}

#' @rdname torch_upsample_nearest3d
torch_upsample_nearest3d <- function(input, self, output_size = NULL,
                                     scale_factors = NULL, scales_d = NULL,
                                     scales_h = NULL, scales_w = NULL) {
  args <- list(
    input = input, output_size = output_size,
    scale_factors = scale_factors, scales_d = scales_d,
    scales_h = scales_h, scales_w = scales_w
  )

  if (!missing(self)) {
    args$self <- self
  }

  do.call(.torch_upsample_nearest3d, args)
}

#' @rdname torch_upsample_nearest3d
torch_upsample_trilinear3d <- function(input, self, output_size = NULL, align_corners,
                                       scale_factors = NULL, scales_d = NULL, scales_h = NULL,
                                       scales_w = NULL) {
  args <- list(
    input = input, output_size = output_size,
    scale_factors = scale_factors, scales_d = scales_d,
    scales_h = scales_h, scales_w = scales_w
  )

  if (!missing(self)) {
    args$self <- self
  }

  if (!missing(align_corners)) {
    args$align_corners <- align_corners
  }

  do.call(.torch_upsample_trilinear3d, args)
}

#' @rdname torch_atleast_1d
torch_atleast_1d <- function(self) {
  if (is_torch_tensor(self)) {
    .torch_atleast_1d(self = self)
  } else {
    .torch_atleast_1d(tensors = self)
  }
}

#' @rdname torch_atleast_2d
torch_atleast_2d <- function(self) {
  if (is_torch_tensor(self)) {
    .torch_atleast_2d(self = self)
  } else {
    .torch_atleast_2d(tensors = self)
  }
}

#' @rdname torch_atleast_3d
torch_atleast_3d <- function(self) {
  if (is_torch_tensor(self)) {
    .torch_atleast_3d(self = self)
  } else {
    .torch_atleast_3d(tensors = self)
  }
}

#' @rdname torch_dequantize
torch_dequantize <- function(tensor) {
  if (is_torch_tensor(tensor)) {
    .torch_dequantize(self = tensor)
  } else {
    .torch_dequantize(tensors = tensor)
  }
}

#' @rdname torch_kaiser_window
torch_kaiser_window <- function(window_length, periodic, beta, dtype = torch_float(),
                                layout = NULL, device = NULL, requires_grad = NULL) {
  options <- torch_tensor_options(
    dtype = dtype, layout = layout, device = device,
    requires_grad = requires_grad
  )
  args <- list(
    window_length = window_length, periodic = periodic,
    options = options
  )

  if (!missing(beta)) {
    args$beta <- beta
  }

  do.call(.torch_kaiser_window, args)
}

#' @rdname torch_vander
torch_vander <- function(x, N = NULL, increasing = FALSE) {
  .torch_vander(x, N, increasing)
}

#' @rdname torch_movedim
torch_movedim <- function(self, source, destination) {
  .torch_movedim(self, as_1_based_dim(source), as_1_based_dim(destination))
}

#' @rdname torch_norm
torch_norm <- function(self, p = 2L, dim, keepdim = FALSE, dtype) {
  if (missing(dtype)) {
    dtype <- self$dtype
  }

  p <- Scalar$new(p)
  if (missing(dim) && !missing(dtype)) {
    o <- cpp_torch_namespace_norm_self_Tensor_p_Scalar_dtype_ScalarType(
      self = self$ptr,
      p = p$ptr,
      dtype = dtype$ptr
    )

    return(Tensor$new(ptr = o))
  }

  if (is.numeric(unlist(dim))) {
    o <- cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType(
      self = self$ptr, p = p$ptr, dim = unlist(dim), keepdim = keepdim, dtype = dtype$ptr
    )
  } else if (is.character(unlist(dim))) {
    o <- cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType(
      self = self$ptr, p = p$ptr, dim = DimnameList$new(unlist(dim))$ptr, keepdim = keepdim, dtype = dtype$ptr
    )
  }

  Tensor$new(ptr = o)
}

torch_one_hot <- function(self, num_classes = -1L) {
  .torch_one_hot(as_1_based_tensor(self), num_classes)
}

#' @rdname torch_split
torch_split <- function(self, split_size, dim = 1L) {
  if (length(split_size) > 1) {
    torch_split_with_sizes(self, split_size, dim)
  } else {
    .torch_split(self, split_size, dim)
  }
}

#' @rdname torch_nonzero
torch_nonzero <- function(self, as_list = FALSE) {
  if (!as_list) {
    out <- .torch_nonzero(self)
    return(out + 1L)
  } else {
    out <- torch_nonzero_numpy(self)
    return(lapply(out, function(x) x + 1L))
  }
}

#' Normal distributed
#'
#' @param mean (tensor or scalar double) Mean of the normal distribution.
#'   If this is a [torch_tensor()] then the output has the same dim as `mean`
#'   and it represents the per-element mean. If it's a scalar value, it's reused
#'   for all elements.
#' @param std (tensor or scalar double) The standard deviation of the normal
#'   distribution. If this is a [torch_tensor()] then the output has the same size as `std`
#'   and it represents the per-element standard deviation. If it's a scalar value,
#'   it's reused for all elements.
#' @param size (integers, optional) only used if both `mean` and `std` are scalars.
#' @param generator a random number generator created with [torch_generator()]. If `NULL`
#'   a default generator is used.
#' @param ... Tensor option parameters like `dtype`, `layout`, and `device`.
#'   Can only be used when `mean` and `std` are both scalar numerics.
#'
#' @rdname torch_normal
#'
#' @export
torch_normal <- function(mean, std, size = NULL, generator = NULL, ...) {
  if (missing(mean)) {
    mean <- 0
  }

  if (missing(std)) {
    std <- 1
  }

  if (!is.null(size)) {
    if (is_torch_tensor(mean) || is_torch_tensor(std)) {
      value_error("size is set, but one of mean or std is not a scalar value.")
    }
  }

  if (!length(list(...)) == 0) {
    if (is_torch_tensor(mean) || is_torch_tensor(std)) {
      value_error("options is set, but one of mean or std is not a scalar value.")
    }
  }

  if (is.null(generator)) {
    generator <- .generator_null
  }

  if (!is_torch_tensor(mean) && !is_torch_tensor(std) && is.null(size)) {
    value_error("size is not set.")
  }

  if (!is.null(size)) {
    if (is.list(size)) size <- unlist(size)
    options <- do.call(torch_tensor_options, list(...))
    return(Tensor$new(ptr = cpp_namespace_normal_double_double(
      mean = mean,
      std = std,
      size = size,
      generator = generator$ptr,
      options = options
    )))
  }

  if (is_torch_tensor(mean) && is_torch_tensor(std)) {
    return(Tensor$new(ptr = cpp_namespace_normal_tensor_tensor(
      mean = mean$ptr,
      std = std$ptr,
      generator = generator$ptr
    )))
  }

  if (is_torch_tensor(mean)) {
    return(Tensor$new(ptr = cpp_namespace_normal_tensor_double(
      mean = mean$ptr,
      std = std,
      generator = generator$ptr
    )))
  }

  if (is_torch_tensor(std)) {
    return(Tensor$new(ptr = cpp_namespace_normal_double_tensor(
      mean = mean,
      std = std$ptr,
      generator = generator$ptr
    )))
  }

  value_error("Please report a bug report in GitHub")
}

#' @rdname torch_polygamma
torch_polygamma <- function(n, input) {
  input <- input$clone()
  input$polygamma_(n = n)
  input
}

#' @rdname torch_fft_fft
torch_fft_fft <- function(self, n = NULL, dim = -1L, norm = NULL) {
  if (is.null(norm)) {
    norm <- "backward"
  }
  .torch_fft_fft(self = self, n = n, dim = dim, norm = norm)
}

#' @rdname torch_fft_ifft
torch_fft_ifft <- function(self, n = NULL, dim = -1L, norm = NULL) {
  if (is.null(norm)) {
    norm <- "backward"
  }
  .torch_fft_ifft(self = self, n = n, dim = dim, norm = norm)
}

#' @rdname torch_fft_rfft
torch_fft_rfft <- function(self, n = NULL, dim = -1L, norm = NULL) {
  if (is.null(norm)) {
    norm <- "backward"
  }
  .torch_fft_rfft(self = self, n = n, dim = dim, norm = norm)
}

#' @rdname torch_fft_irfft
torch_fft_irfft <- function(self, n = NULL, dim = -1L, norm = NULL) {
  if (is.null(norm)) {
    norm <- "backward"
  }
  .torch_fft_irfft(self = self, n = n, dim = dim, norm = norm)
}

torch_broadcast_shapes <- function(...) {
  shapes <- rlang::list2(...)
  with_no_grad({
    scalar <- torch_scalar_tensor(0, device = "cpu")
    tensors <- lapply(shapes, function(shape) scalar$expand(shape))
    out <- torch_broadcast_tensors(tensors)[[1]]$shape
  })
  out
}

#' @rdname torch_multinomial
torch_multinomial <- function(self, num_samples, replacement = FALSE, generator = NULL) {
  r <- .torch_multinomial(self, num_samples, replacement = replacement, generator = generator)
  with_no_grad({
    r$add_(torch_scalar(1L))
  })
  r
}

#' Index torch tensors
#'
#' Helper functions to index tensors.
#'
#' @param self (Tensor) Tensor that will be indexed.
#' @param indices (`List[Tensor]`) List of indices. Indices are torch tensors with
#'   `torch_long()` dtype.
#'
#' @name torch_index
#' @export
NULL

#' In-place version of `torch_index_put`.
#' @name torch_index_put_
#' @inheritParams torch_index
#' @param values (Tensor) values that will be replaced the indexed location. Used
#'   for `torch_index_put` and `torch_index_put_`.
#' @param accumulate (bool) Wether instead of replacing the current values with `values`,
#'   you want to add them.
#' @export
NULL

#' Modify values selected by `indices`.
#' @inheritParams torch_index_put_
#' @name torch_index_put
#' @export
NULL
