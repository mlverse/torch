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

cli_abort <- function(message, ..., env = rlang::caller_env()) {
  cli::cli_abort(gettext_torch(message), ..., .envir = env)
}

cli_inform <- function(message, ..., env = rlang::caller_env(), class = NULL) {
  cli::cli_inform(gettext_torch(message), .envir = env, class = class)
}

gettext_torch <- function(message, ..., domain = "R-torch") {
  rlang::check_dots_empty()
  message_t <- gettext(message, domain = "R-torch")
  names(message_t) <- names(message)
  message_t
}

stop_iteration_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torch")[[1]], .envir = env), class = "stop_iteration_error")
}

inform <- rlang::inform

deprecated <- function(..., env = rlang::caller_env()) {
  rlang::warn(gettext(..., domain = "R-torch")[[1]], class = "deprecated")
}
