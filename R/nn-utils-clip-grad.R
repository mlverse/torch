#' Clips gradient norm of an iterable of parameters.
#'
#' The norm is computed over all gradients together, as if they were
#' concatenated into a single vector. Gradients are modified in-place.
#'
#' @param parameters (IterableTensor or Tensor): an iterable of Tensors or a
#'   single Tensor that will have gradients normalized
#' @param max_norm (float or int): max norm of the gradients
#' @param norm_type (float or int): type of the used p-norm. Can be `Inf` for
#'   infinity norm.
#'
#' @return
#' Total norm of the parameters (viewed as a single vector).
#'
#' @export
nn_utils_clip_grad_norm_ <- function(parameters, max_norm, norm_type = 2) {
  if (is_torch_tensor(parameters)) {
    parameters <- list(parameters)
  }

  parameters <- Filter(function(x) !is_undefined_tensor(x$grad), parameters)

  if (length(parameters) == 0) {
    return(torch_tensor(0))
  }

  device <- parameters[[1]]$grad$device

  if (is.infinite(norm_type)) {
    total_norm <- max(sapply(parameters, function(p) p$grad$detach()$abs()$max()$item()))
    total_norm <- torch_tensor(total_norm, device = device)
  } else {
    total_norm <- torch_norm(torch_stack(lapply(parameters, function(p) {
      torch_norm(p$grad$detach(), norm_type)$to(device = device)
    })), norm_type)
  }

  clip_coef <- max_norm / (total_norm + 1e-6)

  if (clip_coef$item() < 1) {
    lapply(parameters, function(p) {
      p$grad$detach()$mul_(clip_coef$to(device = p$grad$device))
    })
  }

  total_norm
}

#' Clips gradient of an iterable of parameters at specified value.
#'
#' Gradients are modified in-place.
#'
#' @param parameters (Iterable(Tensor) or Tensor): an iterable of Tensors or a
#'   single Tensor that will have gradients normalized
#' @param clip_value (float or int): maximum allowed value of the gradients.
#'
#' @details
#' The gradients are clipped in the range
#' \eqn{\left[\mbox{-clip\_value}, \mbox{clip\_value}\right]}
#'
#' @export
nn_utils_clip_grad_value_ <- function(parameters, clip_value) {
  if (is_torch_tensor(parameters)) {
    parameters <- list(parameters)
  }

  parameters <- Filter(function(x) !is_undefined_tensor(x$grad), parameters)

  for (p in parameters) {
    p$grad$data()$clamp_(min = -clip_value, max = clip_value)
  }

  invisible(NULL)
}
