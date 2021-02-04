#'
#' Given a list of values (possibly containing numbers), returns a list where each
#' value is broadcasted based on the following rules:
#'   - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
#'  - numbers.Number instances (scalars) are upcast to tensors having
#'     the same size and type as the first tensor passed to `values`.  If all the
#'     values are scalars, then they are upcasted to scalar Tensors.
#' 
#'     values (list of `numbers.Number`, `torch.*Tensor` or objects implementing __torch_function__)
#' Raises:
#'     ValueError: if any of the values is not a `numbers.Number` instance,
#'         a `torch.*Tensor` instance, or an instance implementing __torch_function__

broadcast_all <- function(values){
  
  conditions <- 
    sapply(values, function(v){
      inherits(v, "torch_tensor") | inherits(v, "numeric") 
    })
  
  #' TODO: add has_torch_function((v,))
  #' See: https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py
  
  if (!all(conditions))
    value_error('Input arguments must all be instances of numeric,',
                'torch_tensor or objects implementing __torch_function__.')
  
  if (!all(sapply(values, function(v) inherits(v, "torch_tensor")))) {
    .options <- list(dtype = torch_get_default_dtype())
    
    for (v in values) {
      if (inherits(v, "torch_tensor")) {
        .options <- list(
          dtype  = v.dtype,
          device = v.device
        )
        break
      }
    }
    
    new_values <- 
      sapply(values, function(v){
        if (inherits(v, "torch_Tensor"))
          v
        else
          do.call(torch_tensor, c(list(v), .options))
      })
    return(torch_broadcast_tensors(new_values))
    
  }

  torch_broadcast_tensors(values)
}
      
.standard_normal <- function(shape, dtype, device){
  # if torch._C._get_tracing_state():
  #   # [JIT WORKAROUND] lack of support for .normal_()
  #   return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
  #                       torch.ones(shape, dtype=dtype, device=device))
  torch_empty(shape, dtype=dtype, device=device)$normal_()
}






