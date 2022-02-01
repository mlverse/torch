
cpp_method <- function(decl) {
  decl %>%
    glue::glue_data(
"
// [[Rcpp::export]]
{cpp_type(.)} {cpp_method_name(.)} ({cpp_signature(.)}) {{
  {cpp_method_body(.)}
}}

"
    )
}

cpp_namespace <- function(decl) {
  decl %>%
    glue::glue_data(
      "
// [[Rcpp::export]]
{cpp_type(.)} {cpp_namespace_name(.)} ({cpp_signature(.)}) {{
  {cpp_namespace_body(.)}
}}

"
    )
}

cpp_type <- function(decl) {

  if (length(decl$returns) == 0) {

    return("void")

  } else if (length(decl$returns) == 1) {

    returns <- decl$returns[[1]]

    if (returns$dynamic_type == "Tensor")
      return("XPtrTorchTensor")

    if (returns$dynamic_type == "void")
      return("void")

    if (returns$dynamic_type == "bool")
      return("XPtrTorchbool")

    if (returns$dynamic_type == "int64_t")
      return("XPtrTorchint64_t")

    if (returns$dynamic_type == "TensorList")
      return("XPtrTorchTensorList")

    if (returns$dynamic_type == "double")
      return("XPtrTorchdouble")

    if (returns$dynamic_type == "QScheme")
      return("Rcpp::XPtr<XPtrTorchQScheme>")

    if (returns$dynamic_type == "Scalar")
      return("XPtrTorchScalar")

    if (returns$dynamic_type == "ScalarType")
      return("XPtrTorchScalarType")

  } else {
    return("Rcpp::List")
  }

}

cpp_method_name <- function(decl) {
  cpp_function_name(decl, "method")
}

cpp_namespace_name <- function(decl) {
  cpp_function_name(decl, "namespace")
}

dispatch_arguments_from_name <- memoise::memoise(function(name) {
  methods <- memoised_declarations() %>% keep(~.x$name == name)
  get_dispatch_arguments(methods)
})

cpp_function_name <- function(method, type) {

  dispatch_arguments <- dispatch_arguments_from_name(method$name)

  arguments <- method$arguments %>%
    map_chr(~.x$name)

  # reorder and filter
  arguments <- dispatch_arguments[dispatch_arguments %in% arguments]

  arg_types <- list()
  for (nm in arguments) {
    arg_types[[nm]] <- method$arguments %>%
      purrr::keep(~.x$name == nm)  %>%
      purrr::pluck(1) %>%
      purrr::pluck("dynamic_type")
  }

  if (is.null(arg_types))
    return(method$name)

  make_cpp_function_name(method$name, arg_types, type)
}

cpp_parameter_type <- function(argument) {

  if (argument$name %in% c("index", "indices", "dims") &&
      argument$dynamic_type == "Tensor")
  {
    return("XPtrTorchIndexTensor")
  }

  if (argument$name %in% c("dims", "dims_self", "dims_other", "dim") &&
      argument$dynamic_type == "IntArrayRef")
  {
    if (argument$type == "c10::optional<IntArrayRef>") {
      return("XPtrTorchOptionalIndexIntArrayRef")
    } else {
      return("XPtrTorchIndexIntArrayRef")
    }
  }

  if (argument$name %in% c("dim", "dim0", "dim1", "dim2", "start_dim", "end_dim", "index") &&
      argument$dynamic_type == "int64_t") {

    if (argument$type == "c10::optional<int64_t>")
      return("XPtrTorchoptional_index_int64_t")
    else
      return("XPtrTorchindex_int64_t")
  }

  if (argument$name == "indices" &&
      argument$dynamic_type == "TensorList") {
    return("XPtrTorchIndexTensorList")
  }

  if (argument$name == "indices" &&
      argument$dynamic_type == "const c10::List<c10::optional<Tensor>> &") {
    return("XPtrTorchOptionalIndexTensorList")
  }

  if (argument$dynamic_type == "Tensor" &&
      argument$type == "const c10::optional<Tensor> &") {
    return("XPtrTorchOptionalTensor")
  }

  if (argument$dynamic_type == "Tensor") {
    declaration <- "XPtrTorchTensor"
  }

  if (argument$dynamic_type == "bool" && argument$type != "c10::optional<bool>") {
    declaration <- "XPtrTorchbool"
  }

  if (argument$dynamic_type == "bool" && argument$type == "c10::optional<bool>") {
    declaration <- "XPtrTorchoptional_bool"
  }

  if (argument$dynamic_type == "DimnameList" && argument$type != "c10::optional<DimnameList>") {
    declaration <-  "XPtrTorchDimnameList"
  }

  if (argument$dynamic_type == "DimnameList" && argument$type == "c10::optional<DimnameList>") {
    declaration <-  "XPtrTorchOptionalDimnameList"
  }

  if (argument$dynamic_type == "TensorList") {
    declaration <- "XPtrTorchTensorList"
  }

  if (argument$dynamic_type == "IntArrayRef" && argument$type == "c10::optional<IntArrayRef>") {
    declaration <- "XPtrTorchOptionalIntArrayRef"
  }

  if (argument$dynamic_type == "IntArrayRef" && argument$type != "c10::optional<IntArrayRef>") {
    declaration <- "XPtrTorchIntArrayRef"
  }

  if (argument$dynamic_type == "ArrayRef<double>" && argument$type == "c10::optional<ArrayRef<double>>") {
    declaration <- "XPtrTorchOptionalDoubleArrayRef"
  }

  if (argument$dynamic_type == "ArrayRef<double>"&& argument$type != "c10::optional<ArrayRef<double>>") {
    declaration <- "std::vector<double>"
  }

  if (argument$dynamic_type == "int64_t" && !argument$type == "c10::optional<int64_t>") {
    declaration <- "XPtrTorchint64_t"
  }

  if (argument$dynamic_type == "int64_t" && argument$type == "c10::optional<int64_t>") {
    declaration <- "XPtrTorchoptional_int64_t"
  }

  if (argument$dynamic_type == "double" && argument$type == "c10::optional<double>") {
    declaration <- "XPtrTorchOptionaldouble"
  }

  if (argument$dynamic_type == "double" && !argument$type == "c10::optional<double>") {
    declaration <- "XPtrTorchdouble"
  }

  if (argument$dynamic_type == "::std::array<bool,4>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "TensorOptions") {
    declaration <- "XPtrTorchTensorOptions"
  }

  if (argument$dynamic_type == "Generator *") {
    declaration <- "XPtrTorchGenerator"
  }

  if (argument$dynamic_type == "Generator" && argument$type != "c10::optional<Generator>") {
    declaration <- "XPtrTorchGenerator"
  }

  if (argument$dynamic_type == "Generator" && argument$type == "c10::optional<Generator>") {
    declaration <- "XPtrTorchOptionalGenerator"
  }

  if (argument$dynamic_type == "ScalarType" && argument$type != "c10::optional<ScalarType>") {
    declaration <- "XPtrTorchDtype"
  }

  if (argument$dynamic_type == "ScalarType" && argument$type == "c10::optional<ScalarType>") {
    declaration <- "XPtrTorchoptional_scalar_type"
  }

  if (argument$dynamic_type == "Scalar" && !stringr::str_detect(argument$type, "optional")) {
    declaration <- "XPtrTorchScalar"
  }

  if (argument$dynamic_type == "Scalar" && stringr::str_detect(argument$type, "optional")) {
    declaration <- "XPtrTorchoptional_scalar"
  }

  if (argument$dynamic_type == "::std::array<bool,3>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "::std::array<bool,2>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "MemoryFormat" && argument$type != "c10::optional<MemoryFormat>") {
    declaration <- "XPtrTorchMemoryFormat"
  }

  if (argument$dynamic_type == "MemoryFormat" && argument$type == "c10::optional<MemoryFormat>") {
    declaration <- "XPtrTorchoptional_memory_format"
  }

  if (argument$dynamic_type == "std::string" && argument$type != "c10::optional<std::string>") {
    declaration <- "XPtrTorchstring"
  }

  if (argument$dynamic_type == "std::string" && argument$type == "c10::optional<std::string>") {
    declaration <- "XPtrTorchoptional_string"
  }

  if (argument$dynamic_type == "Dimname") {
    declaration <- "XPtrTorchDimname"
  }

  if (argument$dynamic_type == "Device") {
    declaration <- "XPtrTorchDevice"
  }

  if (argument$dynamic_type == "Storage") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "const c10::List<c10::optional<Tensor>> &") {
    declaration <- "XPtrTorchOptionalTensorList"
  }

  if (argument$dynamic_type == "Stream") {
    declaration <- "XPtrTorch"
  }

  if (argument$dynamic_type == "ArrayRef<Scalar>") {
    declaration <- "XPtrTorchvector_Scalar"
  }

  if (argument$dynamic_type == "c10::string_view" && argument$type == "c10::optional<c10::string_view>") {
    declaration <- "XPtrTorchoptional_string_view"
  }

  if (argument$dynamic_type == "c10::string_view" && argument$type != "c10::optional<c10::string_view>") {
    declaration <- "XPtrTorchstring_view"
  }

  # FIXME: Stop if argument$dynamic_type is not handled
  if (!exists("declaration")) {
    stop(paste(argument$dynamic_type, "is not handled!"))
  }

  declaration
}

cpp_parameter_identifier <- function(argument) {

  if (substr(argument$name,1,1) == "_") {
    substr(argument$name, 2, nchar(argument$name))
  } else if (argument$name == "FALSE") {
    'False'
  } else {
    argument$name
  }

}

cpp_parameter <- function(argument) {
  argument %>%
    glue::glue_data("{cpp_parameter_type(.)} {cpp_parameter_identifier(.)}")
}

cpp_signature <- function(decl) {

  res <- purrr::map_chr(decl$arguments, cpp_parameter) %>%
    glue::glue_collapse(sep = ", ")

  if(length(res) == 0)
    res <- ""

  res
}

cpp_argument_transform <- function(argument) {

  argument$name <- cpp_parameter_identifier(argument)

  if (argument$dynamic_type == "Tensor") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "bool") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "DimnameList") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "TensorList") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "IntArrayRef" && argument$type != "c10::optional<IntArrayRef>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "IntArrayRef" && argument$type == "c10::optional<IntArrayRef>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "ArrayRef<double>" && argument$type != "c10::optional<ArrayRef<double>>") {
    result <- glue::glue("lantern_vector_double({argument$name}.data(), {argument$name}.size())")
  }

  if (argument$dynamic_type == "ArrayRef<double>" && argument$type == "c10::optional<ArrayRef<double>>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "int64_t") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "double" && !argument$type == "c10::optional<double>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "double" && argument$type == "c10::optional<double>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "::std::array<bool,4>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "TensorOptions") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Generator *") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Generator") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "ScalarType") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Scalar") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "::std::array<bool,3>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "::std::array<bool,2>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "MemoryFormat") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "std::string") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Dimname") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Device") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Storage") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "const c10::List<c10::optional<Tensor>> &") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "Stream") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "ArrayRef<Scalar>") {
    result <- glue::glue("{argument$name}.get()")
  }

  if (argument$dynamic_type == "c10::string_view") {
    result <- glue::glue("{argument$name}.get()")
  }

  # FIXME: Stop if argument$dynamic_type is not handled
  if (!exists("result")) {
    stop(paste(argument$dynamic_type, "is not handled!"))
  }

  result
}

cast_call <- function(type) {
  function(call) {
    glue::glue("{type}({call})")
  }
}

xptr_return_call <- function(type, dyn_type) {
  function(call) {
    glue::glue("make_xptr<{type}>({call}, \"{dyn_type}\")")
  }
}

cpp_return_statement <- function(returns) {

  if (length(returns) == 1) {

    returns <- returns[[1]]

    if (returns$dynamic_type == "Tensor")
      return(cast_call("XPtrTorchTensor"))

    if (returns$dynamic_type == "bool")
      return(cast_call("XPtrTorchbool"))

    if (returns$dynamic_type == "int64_t")
      return(cast_call("XPtrTorchint64_t"))

    if (returns$dynamic_type == "TensorList")
      return(cast_call("XPtrTorchTensorList"))

    if (returns$dynamic_type == "double")
      return(cast_call("XPtrTorchdouble"))

    if (returns$dynamic_type == "QScheme")
      return(xptr_return_call("XPtrTorchQScheme", "QScheme"))

    if (returns$dynamic_type == "Scalar")
      return(cast_call("XPtrTorchScalar"))

    if (returns$dynamic_type == "ScalarType")
      return(cast_call("XPtrTorchScalarType"))

  } else {

    calls <- purrr::map_chr(
      seq_along(returns),
      ~cpp_return_statement(returns[.x])(glue::glue("lantern_vector_get(wrap.get(), {.x-1})"))
    )

    f <- function(x) {
      glue::glue("Rcpp::List::create({glue::glue_collapse(calls, sep = ',')})")
    }

    return(f)

  }

}

lantern_function_name <- function(method) {

  types <- method$arguments %>%
    purrr::map_chr(~.x$dynamic_type) %>%
    tolower() %>%
    stringr::str_replace_all("[^a-z]", "") %>%
    glue::glue_collapse(sep = "_")

  glue::glue("{tolower(method$name)}_{types}")
}


cpp_method_body <- function(method) {

  arguments <- method$arguments %>%
    purrr::map_chr(cpp_argument_transform)

  if (length(arguments) == 0) {
    arguments <- ""
  } else {
    arguments <- glue::glue_collapse(arguments, sep = ", ")
  }

  method_call <- glue::glue("lantern_Tensor_{lantern_function_name(method)}({arguments});")

  if (length(method$returns) > 0 && method$returns[[1]]$dynamic_type != "void") {
    method_call <- glue::glue("auto r_out = {method_call}")
  }

  if (length(method$returns) > 1) {
    method_call <- c(method_call, "auto wrap = XPtrTorchvector_void(r_out);")
  }

  if (length(method$returns) > 0 && method$returns[[1]]$dynamic_type != "void") {

    return_call <- cpp_return_statement(method$returns)("r_out")
    method_call <- glue::glue_collapse(
      x = c(
        method_call,
        glue::glue("return {return_call};")
      ),
      sep = "\n"
    )

  }

  method_call


}

cpp_namespace_body <- function(method) {

  arguments <- method$arguments %>%
    purrr::map_chr(cpp_argument_transform)

  if (length(arguments) == 0) {
    arguments <- ""
  } else {
    arguments <- glue::glue_collapse(arguments, sep = ", ")
  }

  method_call <- glue::glue("lantern_{lantern_function_name(method)}({arguments});")

  if (length(method$returns) > 0 && method$returns[[1]]$dynamic_type != "void") {
    method_call <- glue::glue("auto r_out = {method_call}")
  }

  if (length(method$returns) > 1) {
    method_call <- c(method_call, "auto wrap = XPtrTorchvector_void(r_out);")
  }

  if (length(method$returns) > 0 && method$returns[[1]]$dynamic_type != "void") {

    return_call <- cpp_return_statement(method$returns)("r_out")
    method_call <- glue::glue_collapse(
      x = c(
        method_call,
        glue::glue("return {return_call};")
      ),
      sep = "\n"
    )

  }

  method_call

}

SKIP_R_BINDIND <- c(
  "set_quantizer_", #https://github.com/pytorch/pytorch/blob/5dfcfeebb89304c1e7978cad7ada1227f19303f6/tools/autograd/gen_python_functions.py#L36
  "normal",
  "polygamma",
  "special_polygamma",
  "_nnpack_available",
  "_backward",
  "_use_cudnn_rnn_flatten_weight",
  "is_vulkan_available",
  "_test_ambiguous_defaults",
  "_test_string_default"
)

SKIP_CPP_BINDING <- c()

cpp <- function(path) {

  decls <-declarations() %>%
    purrr::discard(~.x$name %in% SKIP_R_BINDIND) %>%
    purrr::discard(~.x$name == "range" && length(.x$arguments) == 3)

  pb <- NULL

  methods_code <- decls %>%
    purrr::discard(~.x$name %in% SKIP_CPP_BINDING) %>%
    purrr::keep(~"Tensor" %in% .x$method_of) %>%
    {
      pb <<- progress::progress_bar$new(total = length(.), format = "[:bar] :eta")
      .
    } %>%
    purrr::map_chr(function(x) {
      pb$tick()
      cpp_method(x)
    })

  namespace_code <- decls %>%
    purrr::discard(~.x$name %in% SKIP_CPP_BINDING) %>%
    purrr::keep(~"namespace" %in% .x$method_of) %>%
    {
      pb <<- progress::progress_bar$new(total = length(.), format = "[:bar] :eta")
      .
    } %>%
    purrr::map_chr(function(x) {
      pb$tick()
      cpp_namespace(x)
    })

  writeLines(
    c(
      '// This file is auto generated. Dont modify it by hand.',
      '#include <torch.h>',
      '',
      methods_code,
      namespace_code
    ),
    file.path(path, "/src/gen-namespace.cpp")
  )

}
