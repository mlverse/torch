#' Reads and get's the declaration file.
#'
#' Uses the options `torchgen.version` or `torchgen.path` to get a
#' Declarations.yaml file. If the `torchgen.path` is specified it's
#' always used instead of the `version`.
#'
#' @export
declarations <- function() {

  version <- getOption("torchgen.version", default = "1.10.2")
  path <- getOption("torchgen.path")

  if (is.null(path)) {
    path <- system.file(
      glue::glue("declaration/Declarations-{version}.yaml"),
      package = "torchgen"
    )
  }

  file <- readr::read_file(path)

  new_file <- file %>%
    stringr::str_replace_all(stringr::fixed("at::"), "") %>%
    stringr::str_replace_all(stringr::fixed("const Scalar &"), "Scalar")

  path <- tempfile()
  readr::write_file(new_file, file = path)

  decls <- yaml::read_yaml(
    file = path,
    eval.expr = FALSE,
    handlers = list(
      'bool#yes' = function(x) if (x == "y") x else TRUE,
      'bool#no' = function(x) if (x == "n") x else FALSE,
      int = identity
    )
  )

  # patch declarations for stride to include the int
  index <- which(map_lgl(decls, ~.x$name == "stride"))[1]
  s <- decls[[index]]
  s$method_of <- c(s$method_of, "Tensor")
  decls[[index]] <- s

  decls
}

memoised_declarations <- memoise::memoise(declarations)

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

clean_chars <- c("'", "\"", "%", "#", ":", ">", "<", ",", " ", "*", "&")

clean_names <- function(x) {
  torch:::cpp_clean_names(x, clean_chars)
}

#clean_names <- torch:::clean_names

make_cpp_function_name <- function(method_name, arg_types, type) {

  if (length(arg_types) == 0)
    return(method_name)

  make_cpp_function_name2(method_name, unlist(arg_types), type)
}

make_cpp_function_name2 <- function(method_name, arg_types, type) {
  torch:::cpp_make_function_name(method_name, names(arg_types), arg_types, type)
}
