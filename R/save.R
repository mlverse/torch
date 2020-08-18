#' Saves an object to a disk file.
#' 
#' This function is experimental, don't use for long
#' term storage.
#' 
#' @param obj the saved object
#' @param path a connection or the name of the file to save.
#' @param ... not currently used.
#'
#' @family torch_save
#'
#' @export
torch_save <- function(obj, path, ...) {
  UseMethod("torch_save")
}

#' @export
torch_save.torch_tensor <- function(obj, path, ...) {
  values <- cpp_tensor_save(obj$ptr)
  saveRDS(list(values = values, type = "tensor"), file = path)
  invisible(obj)
}

tensor_to_raw_vector <- function(x) {
  con <- rawConnection(raw(), open = "wr")
  torch_save(x, con)
  r <- rawConnectionValue(con)
  close(con)  
  r
}

#' @export
torch_save.nn_module <- function(obj, path, ...) {
 state_dict <- obj$state_dict()
 state_raw <- lapply(state_dict, tensor_to_raw_vector)
 saveRDS(list(module = obj, state_dict = state_raw, type = "module"), path)
}

#' Loads a saved object
#'
#' @param path a path to the saved object 
#' 
#' @family torch_save
#' 
#' @export
torch_load <- function(path) {
  r <- readRDS(path)
  if (r$type == "tensor")
    torch_load_tensor(r)
  else if (r$type == "module")
    torch_load_module(r)
}

torch_load_tensor <- function(obj) {
  Tensor$new(ptr = cpp_tensor_load(obj$values))
}

torch_load_module <- function(obj) {
  obj$state_dict <- lapply(obj$state_dict, function(x) {
    con <- rawConnection(x)
    r <- readRDS(con)
    close(con)
    torch_load_tensor(r)
  })
  obj$module$load_state_dict(obj$state_dict)
  obj$module
}

load_state_dict <- function(path) {
  path <- normalizePath(path)
  o <- cpp_load_state_dict(path)
  
  values <- TensorList$new(ptr = o$values)$to_r()
  names(values) <- o$keys
  values
}
