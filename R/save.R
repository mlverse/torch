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
#' @concept serialization
#'
#' @export
torch_save <- function(obj, path, ...) {
  UseMethod("torch_save")
}

#' @concept serialization
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

tensor_to_raw_vector_with_class <- function(x) {
  r <- tensor_to_raw_vector(x)
  class(r) <- "torch_serialized_tensor"
  r
}

#' @concept serialization
#' @export
torch_save.nn_module <- function(obj, path, ...) {
  state_dict <- obj$state_dict()
  state_raw <- lapply(state_dict, tensor_to_raw_vector)
  saveRDS(list(module = obj, state_dict = state_raw, type = "module", version = 1), path)
}

#' @concept serialization
#' @export
torch_save.list <- function(obj, path, ...) {
  serialize_tensors <- function(x, f) {
    lapply(x, function(x) {
      if (is_torch_tensor(x)) {
        tensor_to_raw_vector_with_class(x)
      } else if (is.list(x)) {
        serialize_tensors(x)
      } else {
        x
      }
    })
  }

  serialized <- serialize_tensors(obj)
  saveRDS(list(values = serialized, type = "list", version = 1), path)
}

#' Loads a saved object
#'
#' @param path a path to the saved object
#' @param device a device to load tensors to. By default we load to the `cpu` but you can also
#'   load them to any `cuda` device. If `NULL` then the device where the tensor has been saved will
#'   be reused.
#'
#' @family torch_save
#'
#' @export
#' @concept serialization
torch_load <- function(path, device = "cpu") {
  r <- readRDS(path)
  if (r$type == "tensor") {
    torch_load_tensor(r, device)
  } else if (r$type == "module") {
    torch_load_module(r, device)
  } else if (r$type == "list") {
    torch_load_list(r, device)
  }
}

torch_load_tensor <- function(obj, device = NULL) {
  Tensor$new(ptr = cpp_tensor_load(obj$values, device))
}

load_tensor_from_raw <- function(x, device) {
  con <- rawConnection(x)
  r <- readRDS(con)
  close(con)
  torch_load_tensor(r, device)
}

torch_load_module <- function(obj, device = NULL) {
  obj$state_dict <- lapply(obj$state_dict, function(x) {
    load_tensor_from_raw(x, device)
  })

  if (is.null(obj$version) || (obj$version < 1)) {
    obj$module$apply(internal_update_parameters_and_buffers)
  }

  obj$module$load_state_dict(obj$state_dict)
  obj$module
}

torch_load_list <- function(obj, device = NULL) {
  reload <- function(values) {
    lapply(values, function(x) {
      if (inherits(x, "torch_serialized_tensor")) {
        load_tensor_from_raw(x, device = device)
      } else if (is.list(x)) {
        reload(x)
      } else {
        x
      }
    })
  }
  reload(obj$values)
}

#' Load a state dict file
#'
#' This function should only be used to load models saved in python.
#' For it to work correctly you need to use `torch.save` with the flag:
#' `_use_new_zipfile_serialization=True` and also remove all `nn.Parameter`
#' classes from the tensors in the dict.
#'
#' The above might change with development of [this](https://github.com/pytorch/pytorch/issues/37213)
#' in pytorch's C++ api.
#'
#' @param path to the state dict file
#'
#' @return a named list of tensors.
#'
#' @export
#' @concept serialization
load_state_dict <- function(path) {
  path <- normalizePath(path)
  o <- cpp_load_state_dict(path)

  values <- o$values
  names(values) <- o$keys
  values
}

internal_update_parameters_and_buffers <- function(m) {
  to_ptr_tensor <- function(p) {
    if (typeof(p) == "environment") {
      cls <- class(p)
      class(p) <- NULL
      p <- p$ptr
      class(p) <- cls
      p
    }
  }

  # update buffers and params for the new type
  private <- m$.__enclos_env__$private
  for (i in seq_along(private$buffers_)) {
    private$buffers_[[i]] <- to_ptr_tensor(private$buffers_[[i]])
  }
  for (i in seq_along(private$parameters_)) {
    private$parameters_[[i]] <- to_ptr_tensor(private$parameters_[[i]])
  }
}
