library(stringr)
library(purrr)

#
#
# doc <- torch[["mean"]][["__doc__"]]

torch <- reticulate::import("torch")

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

arg_mark <- c("Args:", "Arguments:", "    Arguments:")

get_args <- function(doc) {

  lines <- str_split(doc, "\n")[[1]]
  i <- which(lines %in% arg_mark)

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

  if (lines[i] == "    Arguments:")
    arg_lines <- str_replace_all(arg_lines, "^    ", "")

  l <- str_which(arg_lines, "^    [^ ]+")
  s <- sapply(seq_along(arg_lines), function(x) which.max(l[l<=x]))
  split(arg_lines, s) %>%
    map_chr(~do.call(function(...) str_c(..., collapse = "\n"), as.list(.x))) %>%
    map_chr(str_trim)
}

parse_args <- function(args) {
  x <- str_split_fixed(args, ": ", 2)
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

get_long_desc <- function(doc) {

  lines <- str_split(doc, "\n")[[1]]

  if (any(lines %in% arg_mark)) {
    end <- min(which(lines %in% arg_mark)) -1
  } else if (any(lines == "Example::")) {
    end <- min(which(lines == "Example::")) -1
  } else {
    end <- length(lines)
  }

  str_trim(str_c(lines[1:end], collapse = "\n"))
}

get_signature <- function(doc) {
  desc <- get_long_desc(doc)
  lines <- str_split(desc, "\n")[[1]]

  if (str_detect(lines[1], "->"))
    return(lines[1])
  else
    return(NULL)
}

get_desc <- function(doc) {
  desc <- get_long_desc(doc)
  lines <- str_split(desc, "\n")[[1]]
  if (str_detect(lines[1], "->"))
    return(str_c(lines[-1], collapse = "\n"))
  else
    desc
}

get_examples <- function(doc) {

  lines <- str_split(doc, "\n")[[1]]
  i <- which(lines == "Example::")

  if (length(i) == 0)
    return(NULL)

  if (length(i) > 1)
    stop("cant have more than 1 exzmaples")

  end <- length(lines)

  example_lines <- lines[(i+1):end]
  example_code <- example_lines[str_detect(example_lines, "^    >>>")]
  example_code <- str_replace_all(example_code, "^    >>> ", "")
  example_code <- str_replace_all(example_code, "torch\\.", "torch_")
  str_c(example_code, collapse = "\n")
}

create_roxygen_params <- function(params) {

  if (is.null(params) | length(params) == 0)
    return("#'")

  s <- sapply(params, function(x) {
    glue::glue("#' @param {x$name} {x$type} {x$desc}")
  })
  str_c(s, collapse = "\n")
}

parse_math <- function(desc) {

  lines <- str_split(desc, "\n")[[1]]
  i <- which(lines == ".. math::" | lines == "    .. math::" | lines == ".. math ::")

  if (length(i) == 0)
    return(desc)

  poss <- which(lines == "")

  if(length(poss) > 0) {
    poss <- poss[poss > i[1]]
  }

  if (length(poss) > 0)
    end <- min(poss)
  else
    end <- length(lines) + 1

  lines[i[1]] <- "\\deqn{"
  lines[end] <- "}"

  desc <- str_c(lines, collapse = "\n")

  if (length(i) > 1)
    parse_math(desc)
  else
    desc
}

create_roxygen_desc <- function(desc) {

  if (desc == "" | is.null(desc))
    return("#' Empty description...")

  desc <- str_replace_all(desc, ":attr:", "")
  desc <- str_replace_all(desc, ":func:(`.*`)", "[\\1]")
  desc <- str_replace_all(desc, "`torch\\.(.*)`", "`torch_\\1`")
  desc <- parse_math(desc)
  desc <- str_trim(desc)

  lines <- str_split(desc, "\n")[[1]]
  lines <- str_replace_all(lines, "^.. note::", "@note")
  lines <- str_replace_all(lines, "^.. warning::", "@section Warning:")

  str_c(str_c("#' ", lines), collapse = "\n")
}

create_roxygen_title <- function(name) {
  str_c("#' ", str_to_title(name))
}

create_roxygen_rdname <- function(name) {
  str_c("#' @name torch_", name)
}

create_roxygen_example <- function(exam) {
  if (is.null(exam))
    return("#' ")

  examples <- str_split(exam, "\n")[[1]]
  examples <- str_c("#' ", examples)
  glue::glue(
    "#' @examples",
    "#' \\dontrun{{",
    str_c(examples, collapse = "\n"),
    "#' }}",
    .sep = "\n"
  )
}

create_roxygen_signature_section <- function(sign) {
  if (is.null(sign))
    return("#' ")

  glue::glue(
    "#' @section Signatures:",
    "#' ",
    str_c("#' ", sign),
    "#'",
    .sep = "\n"
  )
}

create_roxygen <- function(name, param, desc, exam, sign) {
  str_c(
    create_roxygen_title(name),
    "#'",
    create_roxygen_desc(desc),
    "#'",
    create_roxygen_signature_section(sign),
    "#'",
    create_roxygen_params(param),
    "#'",
    create_roxygen_example(exam),
    "#'",
    create_roxygen_rdname(name),
    "#'",
    "#' @export",
    "NULL\n",
    sep =  "\n"
  )
}

docum <- function(path) {

  funs <- declarations() %>%
    keep(~"namespace" %in% .x$method_of) %>%
    map_chr(~.x$name) %>%
    unique() %>%
    set_names()

  docs <- map(funs, get_doc) %>% discard(is.null)
  docs <- map(docs, get_signatures)

  args <- map(docs, ~map(.x, . %>% get_args %>% parse_args))
  desc <- map(docs, ~map(.x, get_desc))
  exam <- map(docs, ~map(.x, get_examples))
  sign <- map(docs, ~map(.x, get_signature))

  d <- transpose(list(args = args, desc = desc, exam = exam, sign = sign))
  d <- map(d, transpose)

  out <- imap(d, function(x, name) {
    map(x, ~create_roxygen(name, .x$args, .x$desc, .x$exam, .x$sign))
  })
  out <- map(out, ~str_c(.x, collapse = "\n\n"))
  out <- out[!is.na(out)]
  out <- reduce(out, function(x, y) str_c(x, y, sep = "\n\n"))

  readr::write_file(out, str_c(path, "/R/gen-namespace-docs.R"))
}




