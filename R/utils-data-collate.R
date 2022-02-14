is_tensor_like <- function(x) {
  # we want arrays, matrix's and vectors (lenght > 1) to be converted to
  # tensors. this is only different from torch because there are native
  # scalar types in Python
  has_tensor_like_types <- is.atomic(x) && (!is.character(x))
  has_tensor_like_types && (length(x) > 1 || is.array(x) || is.matrix(x))
}

utils_data_default_collate <- function(batch) {
  elem <- batch[[1]]
  if (is_torch_tensor(elem)) {
    return(torch_stack(batch, dim = 1))
  } else if (is_tensor_like(elem)) {
    # here we rely on tensor like atomic vectors to be cast to torch tensors
    # before stacking.
    return(torch_stack(batch, dim = 1))
  } else if (is.integer(elem) && length(elem) == 1) {
    k <- unlist(batch)
    return(torch_tensor(k, dtype = torch_long()))
  } else if (is.numeric(elem) && length(elem) == 1) {
    k <- unlist(batch)
    return(torch_tensor(k, dtype = torch_float()))
  } else if (is.character(elem) && length(elem) == 1) {
    return(unlist(batch))
  } else if (is.list(elem)) {
    lapply(transpose2(batch), utils_data_default_collate)
  } else {
    value_error("Can't collate data of class: '{class(data)}'")
  }
}

utils_data_default_convert <- function(data) {
  if (is_torch_tensor(data)) {
    return(data)
  } else if (is.list(data)) {
    return(lapply(data, utils_data_default_convert))
  } else {
    tryCatch(
      {
        return(torch_tensor(data))
      },
      error = function(e) {
        value_error("Can't convert data of class: '{class(data)}'")
      }
    )
  }
}
