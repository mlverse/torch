
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

#' Dispatch arguments are those that don't have a default or
#' they occur in multiple signatures with different at least 2 different types.
get_dispatch_arguments <- function(methods) {
  args <- methods %>%
    purrr::map(~.x$arguments) %>%
    purrr::map_dfr(~map_dfr(.x, function(arg) {
     tibble::tibble(
       name = as.character(arg$name),
       type = arg$dynamic_type,
       no_default = is.null(arg$default)
     )
    }))

  args <- args %>%
    dplyr::group_by(name) %>%
    dplyr::filter(
      dplyr::n_distinct(type) > 1 | no_default
    ) %>%
    dplyr::distinct(name) %>%
    dplyr::pull("name")

  args_order <- get_arguments_order(methods)
  args_order[args_order %in% args]
}

#' @importFrom rlang .data
get_arguments_order <- function(methods) {

  order <- methods %>%
    purrr::map(~.x$arguments) %>%
    purrr::map(~purrr::map_chr(.x, ~.x$name)) %>%
    purrr::map_dfr(~tibble::tibble(name = .x, id = seq_along(.x))) %>%
    dplyr::group_by(.data$name) %>%
    dplyr::summarise(id_max = max(.data$id))

  # both have the same id_max but bidirectional should be last.
  if (methods[[1]]$name %in% c("rnn_tanh", "rnn_relu", "lstm", "gru"))
   order$id_max[order$name == "bidirectional"] <- order$id_max[order$name == "bidirectional"] + 100

  # new stable argument is getting placed as first argument with this algorithm
  # which is a breaking change. so we manually move it as last arg.
  if (methods[[1]]$name == "sort")
    order$id_max[order$name == "stable"] <- order$id_max[order$name == "stable"] + 100

  order %>%
    dplyr::arrange(id_max) %>%
    purrr::pluck("name")
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
    purrr::map(~.x$arguments) %>%
    purrr::map_dfr(~purrr::map_dfr(.x, ~tibble(name = .x$name, dtype = .x$dynamic_type))) %>%
    dplyr::group_by(.data$name) %>%
    dplyr::summarise(dtypes = list(unique(.data$dtype))) %>%
    dplyr::mutate(
      c = .data$dtypes %>% purrr::map_chr(~glue::glue("'{.x}'") %>%
                               glue::glue_collapse(sep=",") %>%
                               glue::glue("c(", ., ")")
      )
    ) %>%
    dplyr::mutate(
      c = glue::glue("{name} = {c}")
    )

  glue::glue("expected_types <- list(",
             glue::glue_collapse(dtypes$c, sep = ",\n") ,
             ")"
             )
}

r_namespace <- function(decls) {

  glue::glue("

#' @rdname {r_namespace_name(decls)}
{r_namespace_name(decls)} <- function({r_namespace_signature(decls)}) {{
  {r_namespace_body(decls)}
}}

")
}

creation_ops <- c("ones", "ones_like", "rand", "rand_like", "randint",
                  "randint_like", "randn", "randn_like", "randperm",
                  "zeros", "zeros_like", "arange", "range", "linspace",
                  "logspace", "eye", "empty", "empty_like", "empty_strided",
                  "full", "full_like")

# functions in this list are generated with a preciding '.' in their names so
# wrapers can be defined around them.
internal_funs <- c("logical_not", "max_pool1d_with_indices", "max_pool2d_with_indices",
                   "max_pool2d_with_indices_out", "max_pool3d_with_indices",
                   "max_pool3d_with_indices_out", "max", "min", "max_out", "min_out",
                   "nll_loss", "nll_loss2d", "bartlett_window", "blackman_window",
                   "hamming_window", "hann_window", "normal",
                   "result_type", "sparse_coo_tensor", "stft",
                   "tensordot", "tril_indices", "triu_indices",
                   "multilabel_margin_loss", "multi_margin_loss",
                   "topk", "scalar_tensor", "narrow",
                   "quantize_per_tensor",
                   "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d", "upsample_trilinear3d",
                   "atleast_1d", "atleast_2d", "atleast_3d",
                   "dequantize", "kaiser_window", "vander",
                   "movedim", "argsort", "norm",
                   "argmax", "argmin", "one_hot", "split",
                   "nonzero", "fft_fft", "fft_ifft", "fft_rfft", "fft_irfft",
                   "multinomial", "norm", "cross_entropy_loss", "sort"
                   )

internal_funs <- c(internal_funs, creation_ops)

r_namespace_name <- function(decls) {
  if (decls[[1]]$name %in% internal_funs) {
    glue::glue(".torch_{decls[[1]]$name}")
  } else {
    glue::glue("torch_{decls[[1]]$name}")
  }
}

r_namespace_signature <- function(decls) {
  args <- get_arguments_order(decls)

  if (length(args) == 0)
    return("")

  args %>%
    purrr::map_chr(~r_argument(.x, decls)) %>%
    glue::glue_collapse(sep = ", ")
}

r_argument_default <- function(default) {
  if (default == "c10::nullopt")
    return("NULL")

  if (default == "FALSE")
    return("FALSE")

  if (default == "TRUE")
    return("TRUE")

  if (default %in% c("1", "0", "-1", "2", "-2", "100", "-100",
                     "20")) {
    return(paste0(default, "L"))
  }

  if (default %in% c("0.000010", "0.000000", "0", "2", "-2", "100", "-100", "20",
                     "1.000000", "0.500000", "0.010000", "1", "-1",
                     "10.000000", "0.000001", "0.125000", "0.333333333333333",
                     "0.333333", "9223372036854775807", "1e-05", "1e-08", "0.5", "0.01", "1e-15", "10", "1e-06", "0.125"))
    return(default)

  if (default == "{}")
    return("list()")

  if (default == "MemoryFormat::Contiguous")
    return("torch_contiguous_format()")

  if (default == "MemoryFormContiguous")
    return("torch_contiguous_format()")

  if (default == "nullptr")
    return("NULL")

  if (default == "at::Reduction::Mean")
    return("torch_reduction_mean()")

  if (default == "Reduction::Mean")
    return("torch_reduction_mean()")

  if (default == "{0,1}")
    return("c(0,1)")

  if (default == "{-2,-1}")
    return("c(-2,-1)")

  if (default %in% c("\"L\"", "\"fro\""))
    return(default)

  if (default == "at::kLong")
    return("torch_long()")

  if (default == "kLong")
    return("torch_long()")

  if (default == "\"reduced\"")
    return("\"reduced\"")

  browser()
}

r_argument_name <- function(name) {
  if (name == "FALSE")
    name <- "False"

  if (substr(name, 1, 1) == "_")
    name <- substr(name, 2, nchar(name))

  name
}

r_argument <- function(name, decls) {
  no_default <- get_arguments_with_no_default(decls)
  if (name %in% no_default)
    r_argument_with_no_default(name)
  else
    r_argument_with_default(name, decls)
}


can_be_numeric <- function(x) {
  e <- try(
    x <- eval(parse(text = x)),
    silent = TRUE
  )

  if (inherits(e, "try-error"))
    return(FALSE)

  is.numeric(x)
}

to_1_based <- function(x) {

  x <- eval(parse(text = x))

  is_int <- is.integer(x)

  out <- ifelse(x >= 0, x + 1, x)
  out <- as.character(out)

  if (is_int)
    out <- paste0(out, "L")

  if (length(out) > 1)
    out <- paste0("c(", paste(out, collapse = ","), ")")

  out
}

value_to_char <- function(x) {
  as.character(x)
}


r_argument_with_default <- function(name, decls) {

  default <- purrr::map(decls, ~.x$arguments) %>%
    purrr::flatten() %>%
    purrr::keep(~.x$name == name) %>%
    purrr::map(~.x$default) %>%
    purrr::discard(is.null) %>%
    #purrr::flatten_chr() %>%
    sapply(value_to_char)

  if (length(default) > 1) {
    default <- default[1]
  }

  name <- r_argument_name(name)
  default <- r_argument_default(default)

  # make it 1-based
  if (name %in% c("dim", "dim0", "dim1", "dim2", "start_dim", "end_dim", "dims_self",
                  "dims_other")
      && can_be_numeric(default))
    default <- to_1_based(default)

  glue::glue("{name} = {default}")
}

r_argument_with_no_default <- function(name) {
  r_argument_name(name)
}

r_list_of_arguments_helper <- function(args) {
  if (length(args) == 0)
    return("args <- list()")

  args <- purrr::map_chr(args, r_argument_name)

  args <- glue::glue('"{args}"') %>% glue::glue_collapse(sep = ", ")
  glue::glue("args <- mget(x = c({args}))")
}

r_namespace_list_of_arguments <- function(decls) {
  args <- get_arguments_order(decls)
  r_list_of_arguments_helper(args)
}

r_method_list_of_arguments <- function(decls) {
  args <- get_arguments_order(decls)
  args <- args[args != "self"]
  r_list_of_arguments_helper(args)
}

r_argument_expected_types <- function(arg, decls) {
  decls %>%
    purrr::map(~.x$arguments) %>%
    purrr::flatten() %>%
    purrr::keep(~.x$name == arg) %>%
    purrr::map_chr(~.x$dynamic_type) %>%
    unique()
}

#' @importFrom utils capture.output
r_arguments_expected_types <- function(decls) {
  args <- get_arguments_order(decls)

  if (length(args) == 0) {
    return("expected_types <- list()")
  }

  names(args) <- purrr::map_chr(args, r_argument_name)
  l <- capture.output(
    purrr::map(args, r_argument_expected_types, decls = decls) %>%
      dput()
  )
  glue::glue("expected_types <- {glue::glue_collapse(l, sep = '\n')}")
}

r_arguments_with_no_default <- function(decls) {
  args <- get_dispatch_arguments(decls)
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
  "{r_namespace_list_of_arguments(decls)}",
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

r_method <- function(decls) {

  glue::glue(
  'Tensor$set("{r_method_env(decls)}", "{r_method_name(decls)}", function({r_method_signature(decls)}) {{',
  '  {r_method_body(decls)}',
  '}})'
  )

}

internal_methods <- c("_backward", "retain_grad", "size", "to", "stride",
                      "copy_", "topk", "scatter_", "scatter", "rename",
                      "rename_", "narrow", "narrow_copy", "is_leaf", "max",
                      "min", "argsort", "argmax", "argmin", "norm", "split",
                      "nonzero", "nonzero_numpy", "view", "sort")

r_method_env <- function(decls) {
  if (decls[[1]]$name %in% internal_methods)
    "private"
  else
    "public"
}

r_method_name <- function(decls) {
  name <- decls[[1]]$name
  # if (name == "stride")
  #   browser()

  if (name %in% internal_methods)
    name <- paste0("_", name)

  name
}

r_method_signature <- function(decls) {
  args <- get_arguments_order(decls)
  # the self argument always comes from the Tensor
  args <- args[args != "self"]

  if (length(args) == 0)
    return("")

  args %>%
    purrr::map_chr(~r_argument(.x, decls)) %>%
    glue::glue_collapse(sep = ", ")
}

r_method_body <- function(decls) {

  glue::glue(
    "{r_method_list_of_arguments(decls)}",
    "args <- append(list(self = self), args)",
    "{r_arguments_expected_types(decls)}",
    "{r_arguments_with_no_default(decls)}",
    "{r_return_types(decls)}",
    "call_c_function(",
    "  fun_name = '{decls[[1]]$name}',",
    "  args = args,",
    "  expected_types = expected_types,",
    "  nd_args = nd_args,",
    "  return_types = return_types,",
    "  fun_type = 'method'",
    ")",
    .sep = "\n",
  )

}

r <- function(path) {

  namespace <- declarations() %>%
    purrr::discard(~.x$name %in% SKIP_R_BINDIND[!SKIP_R_BINDIND %in% internal_funs]) %>%
    purrr::keep(~"namespace" %in% .x$method_of)

  namespace_nms <- purrr::map_chr(namespace, ~.x$name)

  namespace_code <- split(namespace, namespace_nms) %>%
    purrr::map(~ .x %>% r_namespace()) %>%
    purrr::flatten_chr()

  namespace_code <- c(
    "# This file is autogenerated. Do not modify it by hand.",
    namespace_code
  )

  writeLines(namespace_code, file.path(path, "/R/gen-namespace.R"))

  methods <- declarations() %>%
    purrr::discard(~.x$name %in% SKIP_R_BINDIND[!SKIP_R_BINDIND %in% internal_methods]) %>%
    purrr::keep(~"Tensor" %in% .x$method_of)

  methods_nms <- purrr::map_chr(methods, ~.x$name)

  methods_code <- split(methods, methods_nms) %>%
    purrr::map(~ .x %>% r_method()) %>%
    purrr::flatten_chr()

  methods_code <- c(
    "# This file is autogenerated. Do not modify by hand.",
    "#' @include tensor.R",
    methods_code
  )

  writeLines(methods_code, file.path(path, "/R/gen-method.R"))

}
