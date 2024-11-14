value_error <- function(..., env = rlang::caller_env()) {
    rlang::abort(glue::glue(gettext(...)[[1]], .envir = env), class = "value_error")
}

type_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "type_error")
}

runtime_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "runtime_error")
}

not_implemented_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "not_implemented_error")
}

warn <- function(..., env = rlang::caller_env()) {
  rlang::warn(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "warning")
}

stop_iteration_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "stop_iteration_error")
}

inform <- rlang::inform

deprecated <- function(..., env = rlang::caller_env()) {
  rlang::warn(gettext(..., domain = "R-torch")[[1]], class = "deprecated")
}
