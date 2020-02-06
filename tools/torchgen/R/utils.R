#' Reads and get's the declaration file.
#'
#' Uses the options `torchgen.version` or `torchgen.path` to get a
#' Declarations.yaml file. If the `torchgen.path` is specified it's
#' always used instead of the `version`.
#'
#' @export
declarations <- function() {

  version <- getOption("torchgen.version", default = "1.4.0")
  path <- getOption("torchgen.path")

  if (is.null(path)) {
    path <- system.file(
      glue::glue("declaration/Declarations-{version}.yaml"),
      package = "torchgen"
    )
  }

  yaml::read_yaml(
    file = path,
    eval.expr = FALSE,
    handlers = list(
      'bool#yes' = function(x) if (x == "y") x else TRUE,
      'bool#no' = function(x) if (x == "n") x else FALSE,
      int = identity
    )
  )

}

#' Get all tensor methods from Declarations.yaml
#'
#' @export
tensor_methods <- memoise::memoise(function() {
  declarations() %>%
    purrr::keep(~"Tensor" %in% .x$method_of)
})

namespace_methods <- memoise::memoise(function() {
  declarations() %>%
    purrr::keep(~"namespace" %in% .x$method_of)
})

#' Takes an object like `declarations()` and
#' returns the list of unique names in it.
#'
#' @param declarations list of methods/function contained
#'  in the declaration.
#'
function_names <- function(declarations) {
  declarations %>%
    purrr::map_chr(~.x$name) %>%
    unique()
}

#' Filters the declarations getting only those with
#' the same name.
#'
#' @inheritParams  function_names
#' @param name name to filter.
#'
declarations_with_name <- function(declarations, name) {
  declarations %>%
    purrr::keep(~.x$name == name)
}

#' Creates a single id for a function.
#'
#' Based on all the arguments.
#
hash_arguments <- function(arguments) {
  types <- paste0(purrr::map_chr(arguments, ~.x$type), collapse = "")
  names <- paste0(purrr::map_chr(arguments, ~.x$name), collapse = "")
  substr(openssl::md5(glue::glue("{types}{names}")), 1,5)
}

clean_names <- function(x) {
  # adapted from janitor::make_clean_names
  x <- gsub("'", "", x)
  x <- gsub("\"", "", x)
  x <- gsub("%", ".percent_", x)
  x <- gsub("#", ".number_", x)
  x <- gsub(":", "", x)
  x <- gsub("<", "", x)
  x <- gsub(">", "", x)
  x <- gsub(",", "", x)
  x <- gsub(" *", "", x, fixed = TRUE)
  x <- gsub("^[[:space:][:punct:]]+", "", x)
  x
}

make_cpp_function_name <- function(method_name, arg_types, type) {

  suffix <- paste(names(arg_types), arg_types, sep = "_")
  suffix <- paste(suffix, collapse = "_")

  if (length(suffix) == 0)
    suffix <- ""

  clean_names(glue::glue("cpp_torch_{type}_{method_name}_{suffix}"))
}
