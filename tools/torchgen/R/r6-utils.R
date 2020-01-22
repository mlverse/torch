#' Returns a named list with default values for all arguments.
#'
#'
get_default_values <- function(methods) {

  arguments <- purrr::map(methods, ~.x$arguments) %>%
    purrr::flatten()

  argument_names <- arguments %>% purrr::map_chr(~.x$name)

  arguments %>% split(argument_names) %>%
    purrr::map_depth(2, ~.x$default %||% NA_character_) %>%
    purrr::map(as.character) %>%
    purrr::map(function(x) {
      dplyr::case_when(
        x == "c10::nullopt" ~ "NULL",
        x == "{}" ~ "NULL",
        x == "{0,1}" ~ "c(0, 1)",
        TRUE ~ x
      )
    }) %>%
    purrr::map(unique)
}

get_return_types <- function(methods) {
  methods %>%
    purrr::map(~.x$returns %>% purrr::map_chr(~.x$type))
}
