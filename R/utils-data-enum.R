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
#' @param ... passed to specific methods.
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
  deprecated(message = c(
    "The `enumerate` construct is deprecated in favor of the `coro::loop` syntax.",
    "See https://github.com/mlverse/torch/issues/558 for more information."
  ))

  if (is.na(length(x))) {
    len <- max_len
  } else {
    len <- length(x)
  }

  # parent environment that only contains the initialized iterator.
  # and will keep the last runned batch.
  p <- rlang::new_environment(list(.iter = x$.iter()))

  # we return a list of environments that have `p` (containing the iterator)
  # as a pointer. All starting with `run = FALSE`. The first time we get an
  # element from this environment `run` will become FALSE and we will no
  # longer need to run the iterator.
  v <- vector(mode = "list", length = len)
  for (i in seq_along(v)) {
    v[[i]] <- new_enum_env(list(run = FALSE), parent = p)
  }

  v
}
