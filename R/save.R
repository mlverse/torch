#' Saves an object to a disk file
#' 
#' @param obj the saved object
#' @param path a path to save the object
#'
#' @export
torch_save <- function(obj, path, ...) {
  UseMethod("torch_save")
}

#' @export
torch_save.torch_tensor <- function(obj, path, ...) {
  values <- as_array(obj)
  meta <- list(dtype = as.character(obj$dtype()))
  saveRDS(list(values = values, meta = meta), file = path)
  invisible(obj)
}

#' Loads a saved object
#'
#' @param path a path to the saved object 
#' 
#' @export
torch_load <- function(path) {
  r <- readRDS(path)
  torch_tensor(r$values, dtype = dtype_from_string(r$meta$dtype))
}