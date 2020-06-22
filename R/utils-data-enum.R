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

#' Dataloader enum
#' 
#' @param dataloader the dataloader to enumerate.
#' @param max_len maximum number of iterations.
#' 
#' @export
enum <- function(dataloader, max_len = 1e6) {
  iter <- dataloader$.iter()
  
  if (is.na(length(dataloader)))
    len <- max_len
  else
    len <- length(dataloader)
  
  p <- rlang::env(.iter = dataloader$.iter())
  
  v <- vector(mode = "list", length = len)
  for (i in seq_along(v)) 
    v[[i]] <- new_enum_env(list(run = FALSE), parent = p)
    
  v
}
