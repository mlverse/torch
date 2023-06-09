#' Saves an object to a disk file.
#'
#' This function is experimental, don't use for long
#' term storage.
#'
#' @param obj the saved object
#' @param path a connection or the name of the file to save.
#' @param ... not currently used.
#' @param compress a logical specifying whether saving to a named file is to use 
#'   "gzip" compression, or one of "gzip", "bzip2" or "xz" to indicate the type of 
#'   compression to be used. Ignored if file is a connection.
#' @family torch_save
#' @concept serialization
#'
#' @export
torch_save <- function(obj, path, ..., compress = TRUE) {
  UseMethod("torch_save")
}

ser_version <- 2
use_ser_version <- function() {
  getOption("torch.serialization_version", ser_version)
}

#' @concept serialization
#' @export
torch_save.torch_tensor <- function(obj, path, ..., compress = TRUE) {
  if (use_ser_version() <= 2) 
    return(legacy_save_torch_tensor(obj, path, ..., compress))
  
  con <- create_write_con(path)
  on.exit({close(con)}, add = TRUE)
  
  safetensors::safe_save_file(
    list("...unnamed..." = obj), 
    path = con,
    metadata = list(
      ..r = list(
        version = use_ser_version(),
        type = "tensor"  
      )
    )
  )
  
  invisible(obj)
}

legacy_save_torch_tensor <- function(obj, path, ..., compress = TRUE) {
  version <- use_ser_version()
  values <- cpp_tensor_save(obj$ptr, base64 = version < 2)
  saveRDS(list(values = values, type = "tensor", version = version), 
          file = path, compress = compress)
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
torch_save.nn_module <- function(obj, path, ..., compress = TRUE) {
  if (use_ser_version() <= 2) 
    return(legacy_save_nn_module(obj, path, ..., compress))
  
  con <- create_write_con(path)
  on.exit({close(con)}, add = TRUE)
  
  metadata = list(..r = list(
    type = "module", 
    version = use_ser_version()
  ))
  
  state_dict <- obj$state_dict()
  
  safetensors::safe_save_file(
    state_dict, 
    path = con, 
    metadata = metadata
  )
  serialize(obj, con = con)
  flush(con)
  
  invisible(obj)
}

legacy_save_nn_module <- function(obj, path, ..., compress = TRUE) {
  state_dict <- obj$state_dict()
  state_raw <- lapply(state_dict, tensor_to_raw_vector)
  saveRDS(list(module = obj, state_dict = state_raw, type = "module", 
               version = use_ser_version()), path, compress = compress)
}

#' @export
torch_save.name <- function(obj, path, ..., compress= TRUE) {
  if (!is_exhausted(obj)) rlang::abort("Cannot save `name` objects.")
  saveRDS(list(type = "coro::exhausted", version = use_ser_version()), path, 
          compress = compress)
}

#' @concept serialization
#' @export
torch_save.list <- function(obj, path, ..., compress = TRUE) {
  if (use_ser_version() <= 2) 
    return(legacy_save_torch_list(obj, path, ..., compress))
  
  lxt <- list_state_dict(obj)
  
  metadata = list(..r = list(
    type = "list", 
    version = use_ser_version()
  ))
  
  con <- create_write_con(path)
  on.exit({close(con)}, add = TRUE)
  
  safetensors::safe_save_file(
    lxt$state_dict, 
    path = con, 
    metadata = metadata
  )
  serialize(lxt$list, con = con)
  flush(con)
  
  invisible(obj)
}

list_state_dict <- function(l, state_dict) {
  miss_dict <- missing(state_dict)
  if (miss_dict) {
    state_dict <- new.env(parent = emptyenv())
  }
  
  l <- lapply(l, function(x) {
    if (inherits(x, "torch_tensor")) {
      addr <- xptr_address(x)
      state_dict[[addr]] <- x
      class(addr) <- "torch_tensor_address"
      addr
    } else if (is.list(x)) {
      list_state_dict(x, state_dict)
    } else {
      x
    }
  })
  
  if (miss_dict) {
    list(list = l, state_dict = as.list(state_dict))  
  } else {
    l
  }
}

list_load_state_dict <- function(l, state_dict) {
  lapply(l, function(x) {
    if (inherits(x, "torch_tensor_address")) {
      state_dict[[x]]
    } else if (is.list(x)) {
      list_load_state_dict(x, state_dict)
    } else {
      x
    }
  })
}

legacy_save_torch_list <- function(obj, path, ..., compress = TRUE) {
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
  saveRDS(list(values = serialized, type = "list", version = use_ser_version()), 
          path, compress = compress)
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
  if (is_rds(path)) {
    return(legacy_torch_load(path, device))
  }
  
  if (!inherits(path, "connection")) {
    con <- create_read_con(path)
    on.exit({close(con)}, add = TRUE)  
  } else {
    con <- path
  }
  
  safe <- safetensors::safe_load_file(con, device = device, framework = "torch")
  meta <- attr(safe, "metadata")[["__metadata__"]][["..r"]]
  
  if (is.null(meta) || is.null(meta$type)) {
    cli::cli_warn(c(
      x = "File not saved with {.fn torch_save}, returning as is.",
      i = "Use {.fn safetensors::safe_load_file} to silence this warning."
    ))
    return(safe)
  }
  
  if (meta$type == "tensor") {
    return(safe[[1]])
  }
  
  max_offset <- attr(safe, "max_offset")
  seek(con, where = max_offset)
  object <- unserialize(con)
  
  if (meta$type == "list") {
    return(list_load_state_dict(object, safe))
  }
  
  if (meta$type == "module") {
    object$load_state_dict(safe, .refer_to_state_dict = TRUE)
    return(object)
  }
  
  stop("currently unsuported")
}

legacy_torch_load <- function(path, device = "cpu") {
  if (is.raw(path)) {
    path <- rawConnection(path)
    on.exit({
      close(path)
    }, add = TRUE)
  }
  
  r <- readRDS(path)
  
  if (!is.null(r$version) && r$version > ser_version) {
    rlang::abort(c(x = paste0(
      "This version of torch can't load files with serialization version > ",
      ser_version)))
  }
  
  if (r$type == "tensor") {
    torch_load_tensor(r, device)
  } else if (r$type == "module") {
    torch_load_module(r, device)
  } else if (r$type == "list") {
    torch_load_list(r, device)
  } else if (r$type == "coro::exhausted") {
    return(coro::exhausted())
  }
}

#' Serialize a torch object returning a raw object
#'
#' It's just a wraper around [torch_save()].
#' 
#' @inheritParams torch_save
#' @param ... Additional arguments passed to [torch_save()]. `obj` and `path` are
#'   not accepted as they are set by [torch_serialize()].
#' @returns A raw vector containing the serialized object. Can be reloaded using
#'   [torch_load()].
#' @family torch_save
#' @concept serialization
torch_serialize <- function(obj, ...) {
  con <- rawConnection(raw(), open = "wr")
  on.exit({
    close(con)
  }, add = TRUE)
  torch_save(obj = obj, path = con, ...)
  rawConnectionValue(con)
}

torch_load_tensor <- function(obj, device = NULL) {
  if (is.null(obj$version) || obj$version < 2) {
    base64 <- TRUE
  } else {
    base64 <- FALSE
  }
  cpp_tensor_load(obj$values, device, base64)
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
#' @param legacy_stream if `TRUE` then the state dict is loaded using a
#'   a legacy way of handling streams.
#' @param ... additional arguments that are currently not used.
#'
#' @return a named list of tensors.
#'
#' @export
#' @concept serialization
load_state_dict <- function(path, ..., legacy_stream = FALSE) {
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

# used to avoid warnings when passing compress by default.
saveRDS <- function(object, file, compress = TRUE) {
  if (compress) {
    base::saveRDS(object, file)
  } else {
    base::saveRDS(object, file, compress = compress)
  }
}

is_rds <- function(con) {
  if (inherits(con, "rawConnection")) {
    on.exit({seek(con, where = 0L)}, add = TRUE)
  } else if (is.raw(con)) {
    con <- rawConnection(con)
    on.exit({close(con)}, add = TRUE)
  }
  
  !inherits(try(readRDS(con), silent = TRUE), "try-error")
}

create_write_con <- function(path) {
  if (is.raw(path)) {
    rawConnection(obj, open = "wb")
  } else if (is.character(path)) {
    file(path, open = "wb")
  } else {
    path # path must be a connection in this case.
  }
}

create_read_con <- function(path) {
  if (is.raw(path)) {
    rawConnection(obj, open = "rb")
  } else if (is.character(path)) {
    file(path, open = "rb")
  } else {
    path # path must be a connection in this case.
  }
}
  