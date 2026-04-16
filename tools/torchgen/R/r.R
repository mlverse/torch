
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
                   "multinomial", "norm", "cross_entropy_loss", "sort",
                   "nll_loss_nd", "bincount", "fft_fftfreq",
                   "where", "repeat_interleave"
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

  if (default == "\"linear\"")
    return("\"linear\"")

  if (default == "\"none\"")
    return("\"none\"")

  if (default == "\"constant\"")
    return("\"constant\"")

  if (default == "\"reflect\"")
    return("\"reflect\"")

  if (default == "::std::nullopt") {
    return("NULL")
  }

  if (default == "c10::MemoryFormContiguous") {
    return("torch_contiguous_format()")
  }

  if (default == '""') {
    return('""')
  }

  if (default == '"right"') {
    return('"right"')
  }
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
    purrr::map_chr(r_mask_dynamic_type_name) %>%
    unique()
}

r_mask_dynamic_type_name <- function(dyn_type) {
  if (dyn_type == "const ITensorListRef &") {
    return("TensorList")
  }

  dyn_type
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

# Dispatch is trivial when there is only one overload, so the resolved
# C++ function name is fully determined at codegen time.
is_trivial_dispatch <- function(decls) {
  length(decls) == 1
}

# Identify the single dispatch arg that has multiple types across overloads.
# Returns NULL if there isn't exactly one such arg.
get_single_multitype_arg <- function(decls) {
  dispatch_args <- get_dispatch_arguments(decls)
  multitype_args <- Filter(function(a) {
    length(r_argument_expected_types(a, decls)) > 1
  }, dispatch_args)
  if (length(multitype_args) == 1) multitype_args[[1]] else NULL
}

# Map a set of expected types to an R expression that checks whether the
# dispatch arg should resolve to a given target type. Returns NULL if we
# don't have a known check for this type pair.
r_type_check <- function(target_type, all_types, arg_r_name) {
  pair <- sort(all_types)
  key <- paste(pair, collapse = "|")

  # Scalar | Tensor — the dominant case (229 functions)
  if (key == "Scalar|Tensor") {
    if (target_type == "Tensor")
      return(glue::glue("is_tensor_dispatch({arg_r_name})"))
    else
      return(glue::glue("!is_tensor_dispatch({arg_r_name})"))
  }

  # Dimname | int64_t
  if (key == "Dimname|int64_t") {
    if (target_type == "Dimname")
      return(glue::glue("is.character({arg_r_name})"))
    else
      return(glue::glue("!is.character({arg_r_name})"))
  }

  # Tensor | double
  if (key == "Tensor|double") {
    if (target_type == "Tensor")
      return(glue::glue("is_tensor_dispatch({arg_r_name})"))
    else
      return(glue::glue("!is_tensor_dispatch({arg_r_name})"))
  }

  # DimnameList | IntArrayRef
  if (key == "DimnameList|IntArrayRef") {
    if (target_type == "DimnameList")
      return(glue::glue("is.character({arg_r_name})"))
    else
      return(glue::glue("!is.character({arg_r_name})"))
  }

  # IntArrayRef | c10::string_view
  if (key == "IntArrayRef|c10::string_view") {
    if (target_type == "c10::string_view")
      return(glue::glue("is.character({arg_r_name})"))
    else
      return(glue::glue("!is.character({arg_r_name})"))
  }

  # int64_t | IntArrayRef
  if (key == "IntArrayRef|int64_t") {
    if (target_type == "int64_t")
      return(glue::glue("is_int64_dispatch({arg_r_name})"))
    else
      return(glue::glue("!is_int64_dispatch({arg_r_name})"))
  }

  # IntArrayRef | Tensor
  if (key == "IntArrayRef|Tensor") {
    if (target_type == "Tensor")
      return(glue::glue("inherits({arg_r_name}, 'torch_tensor')"))
    else
      return(glue::glue("!inherits({arg_r_name}, 'torch_tensor')"))
  }

  # Tensor | TensorList
  if (key == "Tensor|TensorList") {
    if (target_type == "Tensor")
      return(glue::glue("inherits({arg_r_name}, 'torch_tensor')"))
    else
      return(glue::glue("!inherits({arg_r_name}, 'torch_tensor')"))
  }

  # int64_t | Tensor
  if (key == "Tensor|int64_t") {
    if (target_type == "Tensor")
      return(glue::glue("inherits({arg_r_name}, 'torch_tensor')"))
    else
      return(glue::glue("!inherits({arg_r_name}, 'torch_tensor')"))
  }

  # Scalar | c10::string_view
  if (key == "Scalar|c10::string_view") {
    if (target_type == "c10::string_view")
      return(glue::glue("is.character({arg_r_name})"))
    else
      return(glue::glue("!is.character({arg_r_name})"))
  }

  NULL
}

# Try to generate inline if/else dispatch for functions with exactly 2
# overloads where one dispatch arg has multiple types.
# Returns NULL if inline dispatch is not possible.
r_inline_dispatch <- function(decls, fun_type) {
  if (length(decls) != 2) return(NULL)
  multitype_arg <- get_single_multitype_arg(decls)
  if (is.null(multitype_arg)) return(NULL)

  arg_r_name <- r_argument_name(multitype_arg)
  all_types <- r_argument_expected_types(multitype_arg, decls)

  # Group overloads by the type of the multi-type dispatch arg.
  overload_by_type <- list()
  for (decl in decls) {
    arg_obj <- Filter(function(x) x$name == multitype_arg, decl$arguments)
    if (length(arg_obj) == 0) return(NULL)
    dtype <- r_mask_dynamic_type_name(arg_obj[[1]]$dynamic_type)
    overload_by_type[[dtype]] <- decl
  }

  # Generate type check for each overload
  branches <- list()
  for (dtype in names(overload_by_type)) {
    check <- r_type_check(dtype, all_types, arg_r_name)
    if (is.null(check)) return(NULL)  # unsupported type pair, fall back

    decl <- overload_by_type[[dtype]]
    fn_name <- cpp_function_name(decl, fun_type)
    fn_args <- purrr::map_chr(decl$arguments, ~ r_argument_name(.x$name))
    args_str <- glue::glue_collapse(fn_args, sep = ", ")
    branches[[length(branches) + 1]] <- list(
      check = check,
      call = glue::glue("{fn_name}({args_str})")
    )
  }

  if (length(branches) < 2) return(NULL)

  # Build if/else chain: first branch gets `if`, last gets `else`
  lines <- character()
  for (i in seq_along(branches)) {
    if (i == 1) {
      lines <- c(lines, glue::glue("if ({branches[[i]]$check}) {{"))
    } else if (i == length(branches)) {
      lines <- c(lines, "} else {")
    } else {
      lines <- c(lines, glue::glue("}} else if ({branches[[i]]$check}) {{"))
    }
    lines <- c(lines, glue::glue("  {branches[[i]]$call}"))
  }
  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}

# For trivial dispatch, compute the resolved C++ function name at codegen time.
# This reuses the same make_cpp_function_name used by cpp.R.
resolve_trivial_fn_name <- function(decls, fun_type) {
  dispatch_args <- get_dispatch_arguments(decls)
  if (length(dispatch_args) == 0) {
    return(make_cpp_function_name(decls[[1]]$name, list(), fun_type))
  }
  arg_types <- list()
  for (a in dispatch_args) {
    types <- r_argument_expected_types(a, decls)
    arg_types[[r_argument_name(a)]] <- types
  }
  make_cpp_function_name(decls[[1]]$name, arg_types, fun_type)
}

# Get the ordered arg names that the resolved C++ function expects.
# For trivial dispatch all overloads resolve to the same function; we find
# the matching overload and return its full arg list (in declaration order).
resolve_trivial_fn_args <- function(decls, fun_type) {
  fn_name <- resolve_trivial_fn_name(decls, fun_type)
  # Check each overload to find the one whose generated name matches
  for (decl in decls) {
    candidate <- cpp_function_name(decl, fun_type)
    if (candidate == fn_name) {
      return(purrr::map_chr(decl$arguments, ~ r_argument_name(.x$name)))
    }
  }
  # Fallback: use get_arguments_order (should not happen for trivial dispatch)
  purrr::map_chr(get_arguments_order(decls), r_argument_name)
}

r_namespace_body <- function(decls) {

  if (is_trivial_dispatch(decls)) {
    fn_name <- resolve_trivial_fn_name(decls, "namespace")
    fn_args <- resolve_trivial_fn_args(decls, "namespace")
    args_str <- glue::glue_collapse(fn_args, sep = ", ")
    return(glue::glue("{fn_name}({args_str})"))
  }

  inline <- r_inline_dispatch(decls, "namespace")
  if (!is.null(inline)) return(inline)

  # Use C++ dispatcher for remaining multi-overload functions
  all_args <- purrr::map_chr(get_arguments_order(decls), r_argument_name)
  args_mget <- glue::glue('"{all_args}"') %>% glue::glue_collapse(sep = ", ")
  dispatcher <- glue::glue("cpp_torch_dispatch_namespace_{decls[[1]]$name}")
  glue::glue("{dispatcher}(mget(x = c({args_mget})))")

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
                      "nonzero", "nonzero_numpy", "view", "sort", "bincount",
                      "movedim", "clone", "detach", "repeat_interleave", "indices")

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

  if (is_trivial_dispatch(decls)) {
    fn_name <- resolve_trivial_fn_name(decls, "method")
    fn_args <- resolve_trivial_fn_args(decls, "method")
    args_str <- glue::glue_collapse(fn_args, sep = ", ")
    return(glue::glue("{fn_name}({args_str})"))
  }

  inline <- r_inline_dispatch(decls, "method")
  if (!is.null(inline)) return(inline)

  # Use C++ dispatcher for remaining multi-overload methods.
  # Methods need self prepended to the args list.
  all_args <- purrr::map_chr(get_arguments_order(decls), r_argument_name)
  other_args <- all_args[all_args != "self"]
  if (length(other_args) > 0) {
    args_mget <- glue::glue('"{other_args}"') %>% glue::glue_collapse(sep = ", ")
    glue::glue(
      'args <- mget(x = c({args_mget}))\n',
      'args <- c(list(self = self), args)\n',
      'cpp_torch_dispatch_method_{decls[[1]]$name}(args)'
    )
  } else {
    glue::glue('cpp_torch_dispatch_method_{decls[[1]]$name}(list(self = self))')
  }

}

r <- function(path) {

  namespace <- declarations() %>%
    purrr::discard(~.x$name %in% SKIP_R_BINDIND[!SKIP_R_BINDIND %in% internal_funs]) %>%
    purrr::discard(~.x$name == "range" && length(.x$arguments) == 3) %>%
    purrr::discard(~.x$name == "range_out" && length(.x$arguments) == 3) %>%
    purrr::discard(~.x$name == "arange" && length(.x$arguments) == 3) %>%
    purrr::discard(~.x$name == "stft" && length(.x$arguments) == 9) %>%
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
