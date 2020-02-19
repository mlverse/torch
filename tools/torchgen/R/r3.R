
get_arguments_with_no_default <- function(methods) {
  args <- get_arguments_order(methods)
  w_default <- get_arguments_with_default(methods)
  args[!args %in% w_default]
}

get_arguments_with_default <- function(methods) {
  methods %>%
    purrr::map(~.x$arguments) %>%
    purrr::map(~purrr::discard(.x, ~is.null(.x$default))) %>%
    purrr::map(~purrr::map_chr(.x, ~.x$name)) %>%
    purrr::flatten_chr() %>%
    unique()
}

get_arguments_order <- function(methods) {
  methods %>%
    map(~.x$arguments) %>%
    map(~map_chr(.x, ~.x$name)) %>%
    map_dfr(~tibble(name = .x, id = seq_along(.x))) %>%
    group_by(name) %>%
    summarise(id_max = max(id)) %>%
    arrange(id_max) %>%
    with(name)
}

make_argument_list <- function(methods) {

  arguments <- get_arguments_with_no_default(methods)
  helper <- glue::glue("{arguments} = missing({arguments})") %>%
    glue::glue_collapse(sep = ", ")

  glue::glue("args <- c({helper})\nargs <- names(args[!args])")
}

make_all_arguments_list <- function(methods) {

  arguments <- get_arguments_with_default(methods)

  helper <- glue::glue("{arguments} = {arguments}") %>%
    glue::glue_collapse(sep = ", ")

  all_args <- glue::glue("all_args <- list({helper})")
  glue::glue("{all_args}\nfor(nm in args) all_args[[nm]] <- environment()[[nm]]")

}

make_expected_types_list <- function(methods) {

  dtypes <- methods %>%
    map(~.x$arguments) %>%
    map_dfr(~map_dfr(.x, ~tibble(name = .x$name, dtype = .x$dynamic_type))) %>%
    group_by(name) %>%
    summarise(dtypes = list(unique(dtype))) %>%
    mutate(
      c = dtypes %>% map_chr(~glue::glue("'{.x}'") %>%
                               glue::glue_collapse(sep=",") %>%
                               glue::glue("c(", ., ")")
      )
    ) %>%
    mutate(
      c = glue::glue("{name} = {c}")
    )

  glue::glue("expected_types <- list(",
             glue::glue_collapse(dtypes$c, sep = ",\n") ,
             ")"
             )
}

r_namespace <- function(decls) {

  glue::glue("

{r_namespace_name(decls)} <- function({r_namespace_signature(decls)}) {{
  {r_namespace_body(decls)}
}}

")
}

r_namespace_name <- function(decls) {
  glue::glue("torch_{decls[[1]]$name}")
}

r_namespace_signature <- function(decls) {
  args <- get_arguments_order(decls)

  if (length(args) == 0)
    return("")

  args %>%
    purrr::map_chr(~r_namespace_argument(.x, decls)) %>%
    glue::glue_collapse(sep = ", ")
}

r_argument_default <- function(default) {
  if (default == "c10::nullopt")
    return("NULL")

  if (default == "FALSE")
    return("FALSE")

  if (default == "TRUE")
    return("TRUE")

  if (default %in% c("1", "0", "-1", "2", "0.000010", "0.000000",
                     "1.000000", "-2", "0.500000", "100", "0.010000",
                     "10.000000", "-100", "0.000001", "0.125000",
                     "0.333333", "9223372036854775807", "20"))
    return(default)

  if (default == "{}")
    return("list()")

  if (default == "MemoryFormat::Contiguous")
    return("torch_contiguous_format()")

  if (default == "nullptr")
    return("NULL")

  if (default == "at::Reduction::Mean")
    return("torch_reduction_mean()")

  if (default == "{0,1}")
    return("c(0,1)")

  if (default == "at::kLong")
    return("torch_long()")

  browser()
}

r_argument_name <- function(name) {
  if (name == "FALSE")
    name <- "False"

  name
}

r_namespace_argument <- function(name, decls) {
  no_default <- get_arguments_with_no_default(decls)
  if (name %in% no_default)
    r_namespace_argument_with_no_default(name)
  else
    r_namespace_argument_with_default(name, decls)
}

r_namespace_argument_with_default <- function(name, decls) {

  default <- purrr::map(decls, ~.x$arguments) %>%
    purrr::flatten() %>%
    purrr::keep(~.x$name == name) %>%
    purrr::map(~.x$default) %>%
    purrr::discard(is.null) %>%
    purrr::flatten_chr() %>%
    unique()

  if (length(default) > 1) {
    browser()
  }

  name <- r_argument_name(name)
  default <- r_argument_default(default)
  glue::glue("{name} = {default}")
}

r_namespace_argument_with_no_default <- function(name) {
  r_argument_name(name)
}

r_list_of_arguments <- function(decls) {
  args <- get_arguments_order(decls)

  if (length(args) == 0)
    return("args <- list()")

  args <- purrr::map_chr(args, r_argument_name)

  args <- glue::glue('"{args}"') %>% glue::glue_collapse(sep = ", ")
  glue::glue("args <- rlang::env_get_list(nms = c({args}))")
}

r_argument_expected_types <- function(arg, decls) {
  decls %>%
    purrr::map(~.x$arguments) %>%
    purrr::flatten() %>%
    purrr::keep(~.x$name == arg) %>%
    purrr::map_chr(~.x$dynamic_type) %>%
    unique()
}

r_arguments_expected_types <- function(decls) {
  args <- get_arguments_order(decls)
  names(args) <- purrr::map_chr(args, r_argument_name)
  l <- capture.output(
    purrr::map(args, r_argument_expected_types, decls = decls) %>%
      dput()
  )
  glue::glue("expected_types <- {glue::glue_collapse(l, sep = '\n')}")
}

r_arguments_with_no_default <- function(decls) {
  args <- get_arguments_with_no_default(decls)
  args <- purrr::map_chr(args, r_argument_name)
  args <- glue::glue_collapse(capture.output(dput(args)), sep = "\n")
  glue::glue("nd_args <- {args}")
}

r_return_types <- function(decls) {
  types <- decls %>%
    purrr::map_chr(function(decl) {

      if (length(decl$returns) == 0)
        'list("void")'
      else if (length(decl$returns) == 1)
        glue::glue("list('{decl$returns[[1]]$dynamic_type}')")
      else
        glue::glue_collapse(capture.output(purrr::map(decl$returns, ~.x$dynamic_type) %>% dput()))

    }) %>%
    unique()
  glue::glue("return_types <- list({glue::glue_collapse(types, ', ')})")
}

r_namespace_body <- function(decls) {

  glue::glue(.sep = "\n",
  "{r_list_of_arguments(decls)}",
  "{r_arguments_expected_types(decls)}",
  "{r_arguments_with_no_default(decls)}",
  "{r_return_types(decls)}",
  "call_c_function(",
    "fun_name = '{decls[[1]]$name}',",
    "args = args,",
    "expected_types = expected_types,",
    "nd_args = nd_args,",
    "return_types = return_types,",
    "fun_type = 'namespace'",
  ")"
  )

}

r <- function() {

  namespace <- declarations() %>%
    purrr::keep(~"namespace" %in% .x$method_of)

  namespace_nms <- purrr::map_chr(namespace, ~.x$name)

  split(namespace, namespace_nms) %>%
    purrr::map(~ .x %>% r_namespace() %>% styler::style_text()) %>%
    purrr::flatten_chr() %>%
    writeLines("../../R/gen-namespace.R")

}
