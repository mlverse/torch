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

cli_abort <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(...)[[1]], .envir = env)
}

cli_inform <- function(..., env = rlang::caller_env(), class = NULL) {
  cli::cli_inform(gettext(..., domain = "R-torch")[[1]], .envir = env, class = class)
}

stop_iteration_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "stop_iteration_error")
}

inform <- rlang::inform

deprecated <- function(..., env = rlang::caller_env()) {
  rlang::warn(gettext(..., domain = "R-torch")[[1]], class = "deprecated")
}
