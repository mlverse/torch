utils_data_default_collate <- function(batch) {
  elem <- batch[[1]]
  if (is_torch_tensor(elem))
    return(torch_stack(batch, dim = 0))
  else if (is.list(elem)) {
    lapply(seq_along(elem), function(i) {
      utils_data_default_collate(lapply(batch, function(x) x[[i]]))
    })
  } else 
    value_error("Can't collate data of class: '{class(data)}'")
}

utils_data_default_convert <- function(data) {
  if (is_torch_tensor(data))
    return(data)
  else if (is.list(data))
    return(lapply(data, utils_data_default_convert))
  else
    value_error("Can't convert data of class: '{class(data)}'")
}