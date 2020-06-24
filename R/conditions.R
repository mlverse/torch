value_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "value_error")
}

runtime_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(..., .envir = env), class = "runtime_error")
}

not_implemented_error <- function(...) {
  rlang::abort(glue::glue(...), class = "not_implemented_error")
}

warn <- function(...) {
  rlang::warn(glue::glue(...), class = "warning")
}

stop_iteration_error <- function(...) {
  rlang::abort(glue::glue(...), class = "stop_iteration_error")
}

inform <- rlang::inform
