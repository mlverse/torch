torch_docs <- readr::read_file("inst/docs/_torch_docs.py")

parse_add_docstr <- function(torch_docs) {

  torch_docs %>%
    stringi:::stri_match_all(
      dotall= TRUE,
      regex = "add_docstr\\((.*?)\\.format\\(([^\\)]*)\\)\\)"
    ) %>%
    purrr::pluck(1)

}

parse_function_name <- function(x) {
  s <- stringi::stri_split(x, regex = "\n")[[1]][1]
  s <- stringi::stri_replace(s, fixed = ".", replacement = "_")
  s <- stringi::stri_replace(s, fixed = ",", replacement = "")
  s
}

parse_description <- function(x) {
  s <- stringi::stri_split(x, regex = "\n")[[1]]
  first_empty <- min(which(s == ""))
  first_Case <- min(which(stringi::stri_detect(s, regex = "^[A-Z]")))

  start <- min(c(first_empty, first_Case))

  first_.. <- min(which(stringi::stri_detect(s, regex = "(\\.\\. [^f^m])|(^Args:$)|(^Arguments:$)")))

  s <- s[start:(first_.. - 1)]
  s <- stringi::stri_replace_all(s, fixed = ":attr:", replacement = "")
  s
}

x <- torch_docs %>%
  parse_add_docstr()

fnames <- purrr::map_chr(x[,2], parse_function_name)
descs <- purrr::map(x[,2], parse_description)


