utils_data_default_collate <- function(batch) {
  elem <- batch[[1]]
  if (is_torch_tensor(elem)) {
    return(torch_stack(batch, dim = 1))
  } else if ((!is.character(elem) && !is.list(elem)) && is.atomic(elem) &&
    (is.array(elem) || is.matrix(elem) || length(elem) > 1)) {
    # we want arrays, matrix's and vectors (lenght > 1) to be converted to
    # tensors. this is only different fromr torch because there's differentiation
    # between lenght 1 vectors and scalars.
    return(utils_data_default_collate(list_of_tensors(batch)))
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
