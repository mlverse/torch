utils_data_default_collate <- function(batch) {
  elem <- batch[[0]]
  if (is_torch_tensor(elem)) {
    torch_stack(batch, dim = 0)
  }
}