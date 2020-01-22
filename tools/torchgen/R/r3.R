
get_arguments_with_no_default <- function(methods) {
  methods %>%
    map(~.x$arguments) %>%
    map(~keep(.x, ~is.null(.x$default))) %>%
    map(~map_chr(.x, ~.x$name)) %>%
    flatten_chr() %>%
    unique()
}

get_arguments_with_default <- function(methods) {
  methods %>%
    map(~.x$arguments) %>%
    map(~discard(.x, ~is.null(.x$default))) %>%
    map(~map_chr(.x, ~.x$name)) %>%
    flatten_chr() %>%
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














