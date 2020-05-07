get_source <- function(nm, module) {
  inspect <- reticulate::import("inspect")
  er <- try(src <- inspect$getsource(module[[nm]]), silent = TRUE)
  if (inherits(er, "try-error"))
    return("")
  else
    src
}

create_function_boilerplate <- function(name, args, src) {
  a <- map_chr(args, ~.x$name)
  s <- str_split(src, "\n")[[1]]

  s <- str_c("# ", s)
  s <- str_c(s, collapse = "\n")
  glue::glue(
    "nnf_{name} <- function({stringr::str_c(a, collapse = ', ')}) {{",
    "{s}",
    "stop('not implemented')",
    "}}",
    .sep = "\n"
  )
}

docum_functional <- function(boilerplate = FALSE) {

  path <- "../../"

  torch <- reticulate::import("torch")
  builtins <- reticulate::import_builtins()
  nms <- builtins$dir(torch$nn$functional)
  nms <- nms[grepl("^[a-z]", nms)]
  nms <- purrr::set_names(nms)

  docs <- map(nms, ~get_doc(.x, torch$nn$functional)) %>% discard(is.null)
  docs <- map(docs, get_signatures)

  args <- map(docs, function(.x) { map(.x, . %>% get_args %>% parse_args)})
  desc <- map(docs, ~map(.x, get_desc))
  exam <- map(docs, ~map(.x, get_examples))
  sign <- map(docs, ~map(.x, get_signature))

  src <- map(nms, ~get_source(.x, torch$nn$functional))
  src <- src[names(src) %in% names(args)]

  d <- transpose(list(args = args, desc = desc, exam = exam, sign = sign, src = src))
  d <- map(d, transpose)

  out <- imap(d, ~create_roxygen(.y, .x, prefix = "nnf_"))
  out <- reduce(out, function(x, y) str_c(x, y, sep = "\n\n"))

  # rewrite examples
  purrr::iwalk(d, ~create_examples(.y, .x,
                            path = str_c(path, "/R/gen-nn-functional-examples.R"),
                            overwrite = overwrite, prefix = "nnf_"))

  # generate boilerplate code
  if (boilerplate) {
    x <- purrr::imap_chr(d, ~create_function_boilerplate(.y, .x[[1]]$args, .x[[1]]$src))
    readr::write_file(
      str_c(x, collapse = "\n\n"),
      str_c(path, "/R/nn-functional.R")
    )
  }

  readr::write_file(out, str_c(path, "/R/gen-nn-functional-docs.R"))
}
