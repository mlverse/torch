
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

  decls %>% glue::glue_data("

{r_namespace_name(.)} <- function({r_namespace_signature(.)}) {{
  {r_namespace_body(.)}
}}

")
}

r_namespace_name <- function(decls) {
  decls[[1]]$name
}

r_namespace_signature <- function(decls) {
  args <- get_arguments_order(decls)
  args %>%
    purrr::map_chr(~r_namespace_argument(.x, decls)) %>%
    glue::glue_collapse(sep = ", ")
}

r_argument_default <- function(default) {
  if (default == "c10::nullopt")
    return("NULL")

  if (default == "FALSE")
    return("FALSE")


  browser()
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

  default <- r_argument_default(default)
  glue::glue("{name} = {default}")
}

r_namespace_argument_with_no_default <- function(name) {
  name
}

r_list_of_arguments <- function(decls) {
  args <- get_arguments_order(decls)
  args <- glue::glue('"{args}"') %>% glue::glue_collapse(sep = ", ")
  glue::glue("args <- env_get_list(nms = c({args}))")
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
  args <- purrr::set_names(get_arguments_order(decls))
  l <- capture.output(
    purrr::map(args, r_argument_expected_types, decls = decls) %>%
      dput()
  )
  glue::glue("expected_types <- {glue::glue_collapse(l, sep = '\n')}")
}

r_arguments_with_no_default <- function(decls) {
  args <- get_arguments_with_no_default(decls)
  args <- glue::glue_collapse(capture.output(dput(args)), sep = "\n")
  glue::glue("nd_args <- {args}")
}

r_namespace_body <- function(decls) {

  glue::glue("

{r_list_of_arguments(decls)}
{r_arguments_expected_types(decls)}
{r_arguments_with_no_default(decls)}
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][names(args_t[[2]]) %in% nd_args]

")

}

do_call <- function(fun, args) {

  args <- lapply(args, function(x) {
    if (inherits(x, "R6"))
      x$ptr
    else
      x
  })

  do.call(fun, args)
}

torch_mean <- function(self, dim, keepdim = TRUE, dtype) {

  args <- rlang::env_get_list(nms = c("self", "dim", "keepdim", "dtype"))



  # args <- c(self = missing(self), dim = missing(dim))
  # args <- names(args[!args])
  #
  # all_args <- list(dtype = dtype, keepdim = keepdim)
  # for(nm in args) all_args[[nm]] <- environment()[[nm]]
  #
  # expected_types <- list(dim = c('IntArrayRef','DimnameList'),
  #                        dtype = c('ScalarType'),
  #                        keepdim = c('bool'),
  #                        self = c('Tensor'))
  #
  # all_args <- all_arguments_to_torch_type(all_args, expected_types)
  #
  # argument_types <- all_args[[2]][args]
  # all_args <- all_args[[1]]
  # fun <- getNamespace("torch")[[make_cpp_function_name("mean", argument_types)]]
  # res <- do_call(fun, all_args)
  #
  # Tensor$new(ptr = res)
}


#
# declarations() %>%
#    keep(~.x$name == "mean") %>%
#    get_arguments_with_no_default()
#
# declarations() %>%
#    keep(~.x$name == "mean") %>%
#    get_arguments_order()
#
# declarations() %>%
#   keep(~.x$name == "mean") %>%
#   make_all_arguments_list()
#
# declarations() %>%
#   keep(~.x$name == "mean") %>%
#   make_expected_types_list()
#
# declarations() %>%
#   keep(~.x$name == "mean") %>%
#   make_argument_list()














