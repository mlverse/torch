#' Reads and get's the declaration file.
#'
#' Uses the options `torchgen.version` or `torchgen.path` to get a
#' Declarations.yaml file. If the `torchgen.path` is specified it's
#' always used instead of the `version`.
#'
#' @export
declarations <- function() {

  version <- getOption("torchgen.version", default = "1.3.0")
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

  clean_names(sprintf("cpp_torch_%s_%s_%s", type, method_name, suffix))
}
