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
#' A = torch_randn(c(2, 3, 3))
#' torch_lu(A)
#' 
#' @export
torch_lu <- function(A, pivot=TRUE, get_infos=FALSE, out=NULL) {
  # If get_infos is True, then we don't need to check for errors and vice versa
  result <- torch__lu_with_info(A, pivot, get_infos)
  
  if (!is.null(out)) {
    if (!is.list(out))
      stop("argument 'out' must be a list of Tensors.")
    
    if (length(out) - as.integer(get_infos) != 2) {
      stop("expected tuple of ", 2 + as.integer(get_infos), " elements but got ",
           length(out))
    }
    
    for (i in seq_len(out)) {
      out[[i]] <- out[[i]]$resize_as_(result[[i]])$copy_(result[[i]])
    }
    
    return(out)
  }
  
 
  if (get_infos)
    return(result)
  else
    return(result[1:2])
}

torch_logical_not <- function(self) {
  .torch_logical_not(self)
}

#' @rdname torch_bartlett_window
torch_bartlett_window <- function(window_length, periodic=TRUE, dtype=NULL, 
                                  layout=torch_strided(), device=NULL, 
                                  requires_grad=FALSE) {
  opt <- torch_tensor_options(dtype = dtype, layout = layout, device = device,
                              requires_grad = requires_grad)
  .torch_bartlett_window(window_length = window_length, periodic = periodic,
                         options = opt)
}

#' @rdname torch_blackman_window
torch_blackman_window <- function(window_length, periodic=TRUE, dtype=NULL, 
                                  layout=torch_strided(), device=NULL, 
                                  requires_grad=FALSE) {
  opt <- torch_tensor_options(dtype = dtype, layout = layout, device = device,
                              requires_grad = requires_grad)
  .torch_blackman_window(window_length = window_length, periodic = periodic,
                         options = opt)
}

#' @rdname torch_hamming_window
torch_hamming_window <- function(window_length, periodic=TRUE, alpha=0.54, 
                                 beta=0.46, dtype=NULL, layout=torch_strided(), 
                                 device=NULL, requires_grad=FALSE) {
  opt <- torch_tensor_options(dtype = dtype, layout = layout, device = device,
                              requires_grad = requires_grad)
  .torch_hamming_window(window_length = window_length, periodic = periodic, 
                        alpha = alpha, beta = beta, options = opt)
}

#' @rdname torch_hann_window
torch_hann_window <- function(window_length, periodic=TRUE, dtype=NULL, 
                              layout=torch_strided(), device=NULL, 
                              requires_grad=FALSE) {
  opt <- torch_tensor_options(dtype = dtype, layout = layout, device = device,
                              requires_grad = requires_grad)
  .torch_hann_window(window_length = window_length, periodic = periodic, 
                     options = opt)
}

#' @rdname torch_normal
torch_normal <- function(mean, std = 1L, size, generator = NULL) {
  .torch_normal(mean, std, size, generator)
}

#' @rdname torch_result_type
torch_result_type <- function(tensor1, tensor2) {
  
  if (is_torch_tensor(tensor1) && is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_tensor_Tensor_other_Tensor(
      tensor1$ptr, 
      tensor2$ptr
    )
  } else if (is_torch_tensor(tensor1) && !is_torch_tensor(tensor2)) {
    o <- cpp_torch_namespace_result_type_tensor_Tensor_other_Scalar(
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
torch_sparse_coo_tensor <- function(indices, values, size=NULL, dtype=NULL, 
                                    device=NULL, requires_grad=FALSE) {
  opt <- torch_tensor_options(dtype = dtype, device = device, 
                              requires_grad = requires_grad)
  
  if (is.null(size))
    .torch_sparse_coo_tensor(indices, values, options = opt)
  else
    .torch_sparse_coo_tensor(indices, values, size = size, options = opt)
}

#' @rdname torch_stft
torch_stft <- function(input, n_fft, hop_length=NULL, win_length=NULL, 
                       window=NULL, center=TRUE, pad_mode='reflect', 
                       normalized=FALSE, onesided=TRUE) {
  if (center) {
    signal_dim <- input$dim()
    extended_shape <- c(
      rep(2, 3 - signal_dim),
      input$size()
    )
    pad <- as.integer(n_fft %/% 2)
    input <- nnf_pad(input$view(extended_shape), c(pad, pad), pad_mode)
    input <- input$view(utils::tail(input$shape(), signal_dim))
  }
  
  .torch_stft(self = input, n_fft = n_fft, hop_length = hop_length, 
              win_length = win_length, window = window, 
              normalized = normalized, onesided = onesided)
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
    
    if (is_torch_tensor(dims))
      dims <- dims$item()
    
    if (dims < 1)
      runtime_error("tensordot expects dims >= 1, but got {dims}")
    
    dims_a <- seq(from = -dims, to = 0)
    dims_b <- seq(from = 1, to = dims)
  }
  
  .torch_tensordot(a, b, dims_a, dims_b)
}

#' @rdname torch_tril_indices
torch_tril_indices <- function(row, col, offset=0, dtype=torch_long(), 
                               device='cpu', layout=torch_strided()) {
  opt <- torch_tensor_options(dtype = dtype, device = device, layout = layout)
  .torch_tril_indices(row, col, offset, options = opt)
}

#' @rdname torch_triu_indices
torch_triu_indices <- function(row, col, offset=0, dtype=torch_long(), 
                               device='cpu', layout=torch_strided()) {
  opt <- torch_tensor_options(dtype = dtype, device = device, layout = layout)
  .torch_triu_indices(row, col, offset, options = opt)
}

#' @rdname torch_multilabel_margin_loss
torch_multilabel_margin_loss <- function(self, target, reduction = torch_reduction_mean()) {
  .torch_multilabel_margin_loss(self, as_1_based_tensor(target), reduction)
}


