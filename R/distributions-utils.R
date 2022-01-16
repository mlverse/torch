#' Given a list of values (possibly containing numbers), returns a list where each
#' value is broadcasted based on the following rules:
#' @param values List of:
#'  - `torch.*Tensor` instances are broadcasted as per `_broadcasting-semantics`.
#'  -  `numeric` instances (scalars) are upcast to tensors having
#' the same size and type as the first tensor passed to `values`.  If all the
#' values are scalars, then they are upcasted to scalar Tensors.
#' values (list of `numeric`, `torch.*Tensor` or objects implementing __torch_function__)
#' @description
#' Raises value_error: if any of the values is not a `numeric` instance,
#' a `torch.*Tensor` instance, or an instance implementing __torch_function__

broadcast_all <- function(values) {
  conditions <-
    sapply(values, function(v) {
      inherits(v, "torch_tensor") | inherits(v, "numeric")
    })

  #' TODO: add has_torch_function((v,))
  #' See: https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py

  if (!all(conditions)) {
    value_error(
      "Input arguments must all be instances of numeric,",
      "torch_tensor or objects implementing __torch_function__."
    )
  }

  if (!all(sapply(values, function(v) inherits(v, "torch_tensor")))) {
    .options <- list(dtype = torch_get_default_dtype())

    for (v in values) {
      if (inherits(v, "torch_tensor")) {
        .options <- list(
          dtype  = v$dtype,
          device = v$device
        )
        break
      }
    }

    new_values <-
      sapply(values, function(v) {
        if (inherits(v, "torch_tensor")) {
          v
        } else {
          do.call(torch_tensor, c(list(v), .options))
        }
      })
    return(torch_broadcast_tensors(new_values))
  }

  torch_broadcast_tensors(values)
}

.standard_normal <- function(shape, dtype, device) {
  # if torch._C._get_tracing_state():
  #   # [JIT WORKAROUND] lack of support for .normal_()
  #   return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
  #                       torch.ones(shape, dtype=dtype, device=device))
  torch_empty(shape, dtype = dtype, device = device)$normal_()
}

#' Converts a tensor of logits into probabilities. Note that for the
#' binary case, each value denotes log odds, whereas for the
#' multi-dimensional case, the values along the last dimension denote
#' the log probabilities (possibly unnormalized) of the events.
#'
#' @noRd
logits_to_probs <- function(logits, is_binary = FALSE) {
  if (is_binary) {
    return(torch_sigmoid(logits))
  }
  nnf_softmax(logits, dim = -1)
}

clamp_probs <- function(probs) {
  eps <- torch_finfo(probs$dtype)$eps
  probs$clamp(min = eps, max = 1 - eps)
}

#' Converts a tensor of probabilities into logits. For the binary case,
#' this denotes the probability of occurrence of the event indexed by `1`.
#' For the multi-dimensional case, the values along the last dimension
#' denote the probabilities of occurrence of each of the events.
#'
#' @noRd
probs_to_logits <- function(probs, is_binary = FALSE) {
  ps_clamped <- clamp_probs(probs)
  if (is_binary) {
    return(torch_log(ps_clamped) - torch_log1p(-ps_clamped))
  }
  torch_log(ps_clamped)
}
