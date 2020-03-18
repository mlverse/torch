library(stringr)
library(purrr)

# torch <- reticulate::import("torch")
#
# doc <- torch[["mean"]][["__doc__"]]

get_doc <- function(nm) {
  er <- try(doc <- torch[[nm]][["__doc__"]], silent = TRUE)
  if (inherits(er, "try-error"))
    return(NULL)
  else
    doc
}

get_signatures <- function(doc) {
  stringr::str_trim(doc) %>%
    str_split(fixed(".. function::")) %>%
    pluck(1) %>%
    map_chr(str_trim) %>%
    discard(~.x == "")
}

get_args <- function(doc) {

  lines <- str_split(doc, "\n")[[1]]
  i <- which(lines == "Args:" | lines == "Arguments:")

  if (length(i) == 0)
    return(list())

  if (length(i) > 1)
    stop("More than 1 argument sections...")

  idx <- which(lines == "")
  poss <- idx[idx > i]
  if (length(poss) == 0)
    end <- length(lines)
  else
    end <- min(poss) - 1

  arg_lines <- lines[(i+1):(end)]
  l <- str_which(arg_lines, "^    .+")
  s <- sapply(seq_along(arg_lines), function(x) which.max(l[l<=x]))
  split(arg_lines, s) %>%
    map_chr(~do.call(function(...) str_c(..., collapse = "\n"), as.list(.x))) %>%
    map_chr(str_trim)
}

parse_args <- function(args) {
  x <- str_split_fixed(args, ":", 2)
  arg_names <- str_extract(x[,1], "^[^( ]*")
  arg_types <- str_extract(x[,1], "\\([^)]*\\)")
  arg_desc <- str_trim(x[,2])
  transpose(
    list(
      name = arg_names,
      type = arg_types,
      desc = arg_desc
    )
  )
}

get_desc <- function(doc) {

  lines <- str_split(doc, "\n")[[1]]

  if (any(lines == "Args:")) {
    end <- min(which(lines == "Args:" | lines == "Arguments:")) -1
  } else if (any(lines == "Example::")) {
    end <- min(which(lines == "Example::"))
  }

  str_trim(str_c(lines[1:end], collapse = "\n"))
}

create_roxygen_params <- function(params) {

  if (is.null(params))
    return("#'")

  s <- sapply(params, function(x) {
    glue::glue("#' @param {x$name} {x$type} {x$desc}")
  })
  str_c(s, collapse = "\n")
}

create_roxygen_desc <- function(desc) {
  lines <- str_split(desc, "\n")[[1]]
  str_c(str_c("#' ", lines), collapse = "\n")
}

create_roxygen_title <- function(name) {
  str_c("#' ", str_to_title(name))
}

create_roxygen_rdname <- function(name) {
  str_c("#' @name torch_", name)
}

create_roxygen <- function(name, param, desc) {
  str_c(
    create_roxygen_title(name),
    "#'",
    create_roxygen_desc(desc),
    "#'",
    create_roxygen_params(param),
    "#'",
    create_roxygen_rdname(name),
    "#'",
    "#' @export",
    "NULL\n",
    sep =  "\n"
  )
}

docs <- function(path) {
  funs <- declarations() %>%
    keep(~"namespace" %in% .x$method_of) %>%
    map_chr(~.x$name) %>%
    unique() %>%
    set_names()

  docs <- map(funs, get_doc) %>% discard(is.null)
  docs <- map(docs, get_signatures)

  args <- map(docs, ~map(.x, . %>% get_args %>% parse_args))
  desc <- map(docs, ~map(.x, get_desc))

  d <- transpose(list(args = args, desc = desc))
  d <- map(d, transpose)

  out <- imap(d, function(x, name) {
    map(x, ~create_roxygen(name, .x$args, .x$desc))
  })
  out <- map(out, ~str_c(.x, collapse = "\n\n"))
  out <- out[!is.na(out)]
  out <- reduce(out, function(x, y) str_c(x, y, sep = "\n\n"))

  readr::write_file(out, str_c(path, "/R/gen-namespace-docs.R"))
}




