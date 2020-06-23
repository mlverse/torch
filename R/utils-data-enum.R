new_enum_env <- function(data, parent) {
  e <- rlang::new_environment(data, parent)
  class(e) <- "enum_env"
  e
}

#' @export
`$.enum_env` <- function(x, name) {
  x[[name]]
}

#' @export
`[[.enum_env` <- function(x, name, ...) {
  
  if (!rlang::env_get(x, "run")) {
    parent.env(x)[["batch"]] <- parent.env(x)$.iter$.next()
    rlang::env_bind(x, run = TRUE)
  }
  
  parent.env(x)[["batch"]][[name]]
}

#' Enumerate an iterator
#' 
#' @param x the generator to enumerate.
#' 
#' @export
enumerate <- function(x, ...) {
  UseMethod("enumerate")
}


#' Enumerate an iterator
#' 
#' @inheritParams enumerate
#' @param max_len maximum number of iterations.
#' 
#' @export
enumerate.dataloader <- function(x, max_len = 1e6, ...) {
  iter <- x$.iter()
  
  if (is.na(length(x)))
    len <- max_len
  else
    len <- length(x)
  
  p <- rlang::env(.iter = x$.iter())
  
  v <- vector(mode = "list", length = len)
  for (i in seq_along(v)) 
    v[[i]] <- new_enum_env(list(run = FALSE), parent = p)
    
  v
}
