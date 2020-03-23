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

clean_doc <- function(doc) {
  x <- str_split(doc, "\n")[[1]]
  if (any(x == "    Args:" | x == "    Arguments:")) {
   x <- str_replace_all(x, "^    (.*)", "\\1")
  }
  str_c(x, collapse= "\n")
}

arg_mark <- c("Args:", "Arguments:", "    Arguments:", "    Args:")

get_args <- function(doc) {

  doc <- clean_doc(doc)
  lines <- str_split(doc, "\n")[[1]]
  i <- which(lines %in% arg_mark)

  if (length(i) == 0)
    return(list())

  if (length(i) > 1)
    stop("More than 1 argument sections...")

  idx <- which(str_detect(lines, "^[^ ]+"))
  poss <- idx[idx > i]
  if (length(poss) == 0)
    end <- length(lines)
  else
    end <- min(poss) - 1

  arg_lines <- lines[(i+1):(end)]

  if (lines[i] == "    Arguments:" | lines[i] == "    Args:")
    arg_lines <- str_replace_all(arg_lines, "^    ", "")

  l <- str_which(arg_lines, paste0("^", str_extract(arg_lines[1], "^[ ]+"),    "[^ ]+"))
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

  if (str_detect(lines[1], "->") | str_detect(lines[1], "[a-z]*\\("))
    return(lines[1])
  else
    return(NULL)
}

get_desc <- function(doc) {
  desc <- get_long_desc(doc)
  lines <- str_split(desc, "\n")[[1]]
  if (str_detect(lines[1], "->") | str_detect(lines[1], "[a-z]+\\("))
    return(str_c(lines[-1], collapse = "\n"))
  else
    desc
}

fix_torch_creation <- function(exam) {
  str_replace_all(exam, "torch_(randn|zeros|ones){1}\\(([0-9, ]+)\\)", "torch_\\1(c(\\2))")
}

fix_python_lists <- function(exam) {
  exam <- str_replace_all(exam, "\\[([0-9, -j]+)\\]", "c(\\1)")
  exam <- str_replace_all(exam, "\\[([(True),(False) ]+)\\]", "c(\\1)")
  exam <- str_replace_all(exam, "(\\+|\\-) ([0-9]{1})j", "\\1 \\2i")
  exam
}

fix_true_false <- function(exam) {
  exam <- str_replace_all(exam, "True", "TRUE")
  exam <- str_replace_all(exam, "False", "FALSE")
  exam
}

fix_python_tuples <- function(exam) {
  str_replace_all(exam, "([^a-z])\\(([^\\)]+)\\)", "\\1list(\\2)")
}

fix_dtype_function <- function(exam) {
  str_replace_all(exam, "dtype=torch_([^ ^)^,]+)", "dtype=torch_\\1\\(\\)")
}

fix_method_call <- function(exam) {
  str_replace_all(exam, "\\.([a-z]{1})", "$\\1")
}

get_examples <- function(doc) {

  doc <- clean_doc(doc)
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
  example_code <- str_c(example_code, collapse = "\n")
  example_code <- fix_torch_creation(example_code)
  example_code <- fix_python_lists(example_code)
  example_code <- fix_true_false(example_code)
  example_code <- fix_dtype_function(example_code)
  example_code <- fix_method_call(example_code)
  example_code <- fix_python_tuples(example_code)
  example_code
}

create_roxygen_params <- function(params) {

  if (is.null(params) | length(params) == 0)
    return("#'")

  s <- sapply(params, function(x) {
    x$desc <- inline_math(x$desc)
    x$desc <- function_reference(x$desc)
    x$desc <- remove_directives(x$desc)
    x$desc <- remove_reference(x$desc)
    x$type <- remove_directives(x$type)
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
    poss <- poss[poss > (i[1] + 1)]
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

inline_math <- function(x) {
  str_replace_all(x, ":math:`([^`]+)`", "\\\\eqn{\\1}")
}

function_reference <- function(x) {
  x <- str_replace_all(x, ":func:`([^`]+)`", "[`\\1`]")
  str_replace_all(x, "`torch\\.(.*)`", "`torch_\\1`")
}

remove_directives <- function(x) {
  x <- str_replace_all(x, ":class:", "")
  x <- str_replace_all(x, ":sup:", "")
  x <- str_replace_all(x, ":attr:", "")
  x <- str_replace_all(x, ":code:", "")
  x <- str_replace_all(x, ":meth:", "")
  x
}

remove_reference <- function(x) {
  str_replace_all(x, ":ref:`([^<^`]+)[^`]*`", "\\1")
}

desc_prep <- function(desc) {
  if (desc == "" | is.null(desc))
    return("#' Empty description...")

  desc <- str_replace_all(desc, ":attr:", "")
  desc <- inline_math(desc)
  desc <- function_reference(desc)
  desc <- parse_math(desc)
  desc <- str_trim(desc)
  desc <- remove_directives(desc)
  desc <- remove_reference(desc)

  lines <- str_split(desc, "\n")[[1]]
  lines <- str_replace_all(lines, "^.. note::", "@note")
  lines <- str_replace_all(lines, "^.. warning::", "@section Warning:")

  str_c(str_c("#' ", lines), collapse = "\n")
}

create_roxygen_desc <- function(desc) {
  desc_prep(desc)
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
    str_c(examples, collapse = "\n"),
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

create_roxygen <- function(name, m) {
  str_c(
    create_roxygen_title(name),
    "#'",
    create_roxygen_full_desc(m),
    "#'",
    create_roxygen_full_params(m),
    "#'",
    create_roxygen_rdname(name),
    "#'",
    "#' @export",
    "NULL\n",
    sep =  "\n"
  )
}

create_examples <- function(name, m) {
  examples <- create_roxygen_full_examples(m)
  hash <- openssl::md5(examples, "md5")

  str_c(
    glue::glue("# -> {name}: {hash} <-"),
    "#'",
    create_roxygen_rdname(name),
    "#'",
    create_roxygen_full_examples(m),
    "NULL",
    sep = "\n"
  )
}

create_roxygen_desc_section <- function(desc, sign) {

  if (is.null(sign))
    sign <- "TEST"

  str_c(
    str_c("#' @section ", sign, " :"),
    "#'",
    desc_prep(desc),
    "#'",
    sep = "\n"
  )
}

create_roxygen_full_desc <- function(m) {
  desc <- map_chr(m, ~create_roxygen_desc_section(.x$desc, .x$sign))
  str_c(desc, collapse = "\n")
}

create_roxygen_full_params <- function(m) {
  pars <- flatten(map(m, ~.x$args))
  pars <- pars[!duplicated(map_chr(pars, ~.x$name))]
  create_roxygen_params(pars)
}

create_roxygen_full_examples <- function(m) {
  examples <- map_chr(m, ~create_roxygen_example(.x$exam))
  str_c("#' @examples\n#'\n", str_c(examples, collapse = "\n#'\n#'\n"))
}

docum <- function(path) {

  funs <- declarations() %>%
    keep(~"namespace" %in% .x$method_of) %>%
    map_chr(~.x$name) %>%
    unique() %>%
    set_names()

  docs <- map(funs, get_doc) %>% discard(is.null)
  docs <- map(docs, get_signatures)

  args <- map(docs, function(.x) { map(.x, . %>% get_args %>% parse_args)})
  desc <- map(docs, ~map(.x, get_desc))
  exam <- map(docs, ~map(.x, get_examples))
  sign <- map(docs, ~map(.x, get_signature))

  d <- transpose(list(args = args, desc = desc, exam = exam, sign = sign))
  d <- map(d, transpose)

  out <- imap(d, ~create_roxygen(.y, .x))
  out <- reduce(out, function(x, y) str_c(x, y, sep = "\n\n"))
  examples <- imap(d, ~create_examples(.y, .x)) %>%
    reduce(~ str_c(.x, .y, sep = "\n\n"))

  readr::write_file(out, str_c(path, "/R/gen-namespace-docs.R"))
  readr::write_file(examples, str_c(path, "/R/gen-namespace-examples.R"))
}




