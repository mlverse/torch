value_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "value_error")
}

type_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "type_error")
}

runtime_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "runtime_error")
}

not_implemented_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "not_implemented_error")
}

warn <- function(..., env = rlang::caller_env()) {
  rlang::warn(glue::glue(..., .envir = env), class = "warning")
}

stop_iteration_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "stop_iteration_error")
}

inform <- rlang::inform

deprecated <- function(..., env = rlang::caller_env()) {
  rlang::warn(..., class = "deprecated")
}
