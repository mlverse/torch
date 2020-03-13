
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
      return("Rcpp::XPtr<XPtrTorchTensor>")

    if (returns$dynamic_type == "void")
      return("void")

    if (returns$dynamic_type == "bool")
      return("bool")

    if (returns$dynamic_type == "int64_t")
      return("int64_t")

    if (returns$dynamic_type == "TensorList")
      return("Rcpp::XPtr<XPtrTorchTensorList>")

    if (returns$dynamic_type == "double")
      return("double")

    if (returns$dynamic_type == "QScheme")
      return("Rcpp::XPtr<XPtrTorchQScheme>")

    if (returns$dynamic_type == "Scalar")
      return("Rcpp::XPtr<XPtrTorchScalar>")

    if (returns$dynamic_type == "ScalarType")
      return("Rcpp::XPtr<XPtrTorchScalarType>")

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

cpp_function_name <- function(method, type) {
  arguments <- get_arguments_with_no_default(list(method))
  arg_types <- list()
  for (nm in arguments) {
    arg_types[[nm]] <- method$arguments %>%
      purrr::keep(~.x$name == nm)  %>%
      purrr::pluck(1) %>%
      purrr::pluck("dynamic_type")
  }

  make_cpp_function_name(method$name, arg_types, type)
}

cpp_parameter_type <- function(argument) {

  if (argument$dynamic_type == "Tensor") {
    declaration <- "Rcpp::XPtr<XPtrTorchTensor>"
  }

  if (argument$dynamic_type == "bool") {
    declaration <- "bool"
  }

  if (argument$dynamic_type == "DimnameList") {
    declaration <-  "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "TensorList") {
    declaration <- "Rcpp::XPtr<XPtrTorchTensorList>"
  }

  if (argument$dynamic_type == "IntArrayRef") {
    declaration <- "std::vector<int64_t>"
  }

  if (argument$dynamic_type == "int64_t") {
    declaration <- "int64_t"
  }

  if (argument$dynamic_type == "double") {
    declaration <- "double"
  }

  if (argument$dynamic_type == "std::array<bool,4>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "TensorOptions") {
    declaration <- "Rcpp::XPtr<XPtrTorchTensorOptions>"
  }

  if (argument$dynamic_type == "Generator *") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "ScalarType") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "Scalar") {
    declaration <- "Rcpp::XPtr<XPtrTorchScalar>"
  }

  if (argument$dynamic_type == "std::array<bool,3>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "std::array<bool,2>") {
    declaration <- "std::vector<bool>"
  }

  if (argument$dynamic_type == "MemoryFormat") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "std::string") {
    declaration <- "std::string"
  }

  if (argument$dynamic_type == "Dimname") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "Device") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
  }

  if (argument$dynamic_type == "Storage") {
    declaration <- "Rcpp::XPtr<XPtrTorch>"
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
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "bool") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "DimnameList") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "TensorList") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "IntArrayRef") {
    result <- glue::glue("lantern_vector_int64_t(&{argument$name}[0], {argument$name}.size())")
  }

  if (argument$dynamic_type == "int64_t") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "double") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "std::array<bool,4>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "TensorOptions") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "Generator *") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "ScalarType") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "Scalar") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "std::array<bool,3>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "std::array<bool,2>") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "MemoryFormat") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "std::string") {
    result <- glue::glue("reinterpret_cast<void*>(&{argument$name})")
  }

  if (argument$dynamic_type == "Dimname") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "Device") {
    result <- glue::glue("{argument$name}->get()")
  }

  if (argument$dynamic_type == "Storage") {
    result <- glue::glue("{argument$name}->get()")
  }

  result
}

xptr_return_call <- function(type, dyn_type) {
  function(call) {
    glue::glue("make_xptr<{type}>({call}, \"{dyn_type}\")")
  }
}

reinterpret_cast_call <- function(dyn_type) {
  function(call) {
    glue::glue("*reinterpret_cast<{dyn_type} *>({call})")
  }
}

cpp_return_statement <- function(returns) {

  if (length(returns) == 1) {

    returns <- returns[[1]]

    if (returns$dynamic_type == "Tensor")
      return(xptr_return_call("XPtrTorchTensor", "Tensor"))

    if (returns$dynamic_type == "bool")
      return(reinterpret_cast_call("bool"))

    if (returns$dynamic_type == "int64_t")
      return(reinterpret_cast_call("int64_t"))

    if (returns$dynamic_type == "TensorList")
      return(xptr_return_call("XPtrTorchTensorList", "TensorList"))

    if (returns$dynamic_type == "double")
      return(reinterpret_cast_call("double"))

    if (returns$dynamic_type == "QScheme")
      return(xptr_return_call("XPtrTorchQScheme", "QScheme"))

    if (returns$dynamic_type == "Scalar")
      return(xptr_return_call("XPtrTorchScalar", "Scalar"))

    if (returns$dynamic_type == "ScalarType")
      return(xptr_return_call("XPtrTorchScalarType", "ScalarType"))

  } else {

    calls <- purrr::map_chr(
      seq_along(returns),
      ~cpp_return_statement(returns[.x])(glue::glue("lantern_vector_get(r_out, {.x-1})"))
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
  "_nnpack_available"
)

cpp <- function(path) {

  decls <-declarations() %>%
    purrr::discard(~.x$name %in% SKIP_R_BINDIND) %>%
    purrr::discard(~.x$name == "range" && length(.x$arguments) == 3)

  methods_code <- decls %>%
    purrr::keep(~"Tensor" %in% .x$method_of) %>%
    purrr::map_chr(cpp_method)

  namespace_code <- decls %>%
    purrr::keep(~"namespace" %in% .x$method_of) %>%
    purrr::map_chr(cpp_namespace)

  writeLines(
    c(
      '// This file is auto generated. Dont modify it by hand.',
      '#include "utils.hpp"',
      '',
      methods_code,
      namespace_code
    ),
    file.path(path, "/src/gen-namespace.cpp")
  )

}
