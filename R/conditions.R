value_error <- function(...) {
  rlang::abort(glue::glue(...), class = "value_error")
}

not_implemented_error <- function(...) {
  rlang::abort(glue::glue(...), class = "not_implemented_error")
}

warn <- function(...) {
  rlang::warn(glue::glue(...), class = "warning")
}
