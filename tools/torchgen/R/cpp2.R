simple_methods <- function() {
  m_names <- declarations() %>%
    purrr:::map_dfr(~tibble::tibble(name = .x$name)) %>%
    dplyr::count(name) %>%
    dplyr::filter(n == 1)
  declarations() %>%
    purrr::keep(~.x$name %in% m_names$name) %>%
    purrr::keep(~any(.x$method_of == "Tensor"))
}

cpp_return_type <- function(method) {

  if (length(method$returns) == 1) {

    returns <- method$returns[[1]]

    if (returns$dynamic_type == "Tensor")
      return("Rcpp:XPtr<torch::Tensor>")

    if (returns$dynamic_type == "void")
      return("void")

    if (returns$dynamic_type == "bool")
      return("bool")

    if (returns$dynamic_type == "int64_t")
      return("int64_t")

    if (returns$dynamic_type == "TensorList")
      return("Rcpp::XPtr<torch::TensorList>")

    if (returns$dynamic_type == "double")
      return("double")

    if (returns$dynamic_type == "QScheme")
      return("Rcpp::XPtr<torch::QScheme>")

    if (returns$dynamic_type == "Scalar")
      return("Rcpp::XPtr<torch::Scalar>")

    if (returns$dynamic_type == "ScalarType")
      return("Rcpp::XPtr<torch::Dtype>")

  } else {
    return("Rcpp::List")
  }

}

clean_names <- function(x) {
  # adapted from janitor::make_clean_names
  x <- gsub("'", "", x)
  x <- gsub("\"", "", x)
  x <- gsub("%", ".percent_", x)
  x <- gsub("#", ".number_", x)
  x <- gsub("^[[:space:][:punct:]]+", "", x)
  x
}

make_cpp_function_name <- function(method_name, argument_types) {

  suffix <- paste(names(arg_types), arg_types, sep = "_")
  suffix <- paste(suffix, collapse = "_")

  if (length(suffix) == 0)
    suffix <- ""

  clean_names(glue::glue("cpp_torch_{method$name}_{suffix}"))
}

cpp_function_name <- function(method) {
  arguments <- get_arguments_with_no_default(list(method))
  arg_types <- list()
  for (nm in arguments) {
    arg_types[[nm]] <- method$arguments %>%
      keep(~.x$name == nm)  %>%
      pluck(1) %>%
      with(dynamic_type)
  }

  make_cpp_function_name(method$name, arg_types)
}

cpp_argument <- function(argument) {

  # declaration

  declaration <- NULL

  if (argument$dynamic_type == "Tensor") {
    declaration <- "Rcpp::XPtr<torch::Tensor>"
  }

  if (argument$dynamic_type == "bool") {
    declaration <- "bool"
  }

  if (argument$dynamic_type == "DimnameList") {
    declaration <-  "Rcpp::XPtr<std::vector<torch::Dimname>>"
  }

  if (argument$dynamic_type == "TensorList") {
    declaration <- "Rcpp::XPtr<std::vector<torch::Tensor>>"
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
    declaration <- "std::array<bool,4>"
  }

  if (argument$dynamic_type == "TensorOptions") {
    declaration <- "Rcpp::XPtr<torch::TensorOptions>"
  }

  if (argument$dynamic_type == "Generator *") {
    declaration <- "Rcpp::XPtr<Generator *>"
  }

  if (argument$dynamic_type == "ScalarType") {
    declaration <- "Rcpp::XPtr<torch::Dtype>"
  }

  if (argument$dynamic_type == "Scalar") {
    declaration <- "Rcpp::XPtr<torch::Scalar>"
  }

  if (argument$dynamic_type == "std::array<bool,3>") {
    declaration <- "std::array<bool,3>"
  }

  if (argument$dynamic_type == "std::array<bool,2>") {
    declaration <- "std::array<bool,2>"
  }

  if (argument$dynamic_type == "MemoryFormat") {
    declaration <- "Rcpp::XPtr<torch::MemoryFormat>"
  }

  if (argument$dynamic_type == "std::string") {
    declaration <- "std::String"
  }

  if (argument$dynamic_type == "Dimname") {
    declaration <- "Rcpp::XPtr<torch::Dimname>"
  }

  if (argument$dynamic_type == "Device") {
    declaration <- "Rcpp::XPtr<torch::Device>"
  }

  if (argument$dynamic_type == "Storage") {
    declaration <- "Rcpp::XPtr<torch::Storage>"
  }

  if (is.null(declaration))
    browser()

  glue::glue("{declaration} {argument$name}")
}

cpp_signature <- function(method) {

  res <- purrr::map_chr(method$arguments, cpp_argument) %>%
    glue::glue_collapse(sep = ", ")

  if(length(res) == 0)
    res <- ""

  res
}

cpp_argument_transform <- function(argument) {

  if (argument$dynamic_type == "Tensor") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "bool") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "DimnameList") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "TensorList") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "IntArrayRef") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "int64_t") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "double") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "std::array<bool,4>") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "TensorOptions") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "Generator *") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "ScalarType") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "Scalar") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "std::array<bool,3>") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "std::array<bool,2>") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "MemoryFormat") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "std::string") {
    result <- glue::glue("{argument$name}")
  }

  if (argument$dynamic_type == "Dimname") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "Device") {
    result <- glue::glue("* {argument$name}")
  }

  if (argument$dynamic_type == "Storage") {
    result <- glue::glue("* {argument$name}")
  }

  result
}

SKIP_R_BINDIND <- c(
  "set_quantizer_" #https://github.com/pytorch/pytorch/blob/5dfcfeebb89304c1e7978cad7ada1227f19303f6/tools/autograd/gen_python_functions.py#L36
)

xptr_return_call <- function(type) {
  function(call) {
    glue::glue("make_xptr<{type}>({call})")
  }
}

cpp_code_return <- function(returns) {

  if (length(returns) == 1) {

    returns <- returns[[1]]

    if (returns$dynamic_type == "Tensor")
      return(xptr_return_call("torch::Tensor"))

    if (returns$dynamic_type == "bool")
      return(identity)

    if (returns$dynamic_type == "int64_t")
      return(identity)

    if (returns$dynamic_type == "TensorList")
      return(xptr_return_call("torch::TensorList"))

    if (returns$dynamic_type == "double")
      return(identity)

    if (returns$dynamic_type == "QScheme")
      return(xptr_return_call("torch::QScheme"))

    if (returns$dynamic_type == "Scalar")
      return(xptr_return_call("torch::Scalar"))

    if (returns$dynamic_type == "ScalarType")
      return(xptr_return_call("torch::Dtype"))

  } else {


    calls <- map_chr(
      seq_along(returns),
      ~cpp_code_return(returns[.x])(glue::glue("std::get<{.x-1}>(out)"))
    )

    f <- function(x) {
      glue::glue("Rcpp::List::create({glue::glue_collapse(calls, sep = ',')})")
    }

    return(f)

  }

  browser()

}

cpp_code <- function(method) {

  arguments <- method$arguments %>%
    discard(~.x$name == "self") %>%
    map_chr(cpp_argument_transform)

  if (length(arguments) == 0) {
    arguments <- ""
  } else {
    arguments <- glue::glue_collapse(arguments, sep = ", ")
  }

  method_call <- glue::glue("self->{method$name}({arguments});")

  if (method$returns[[1]]$dynamic_type != "void") {
    method_call <- glue::glue("auto out = {method_call}")
  }

  if (method$returns[[1]]$dynamic_type != "void") {

    return_call <- cpp_code_return(method$returns)("out")
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

# cpp_return_type(method)
# cpp_function_name(method)
# cpp_argument(method$arguments[[1]])
# cpp_signature(method)
# cpp_code(method)
#
# code <- declarations() %>%
#   purrr::discard(~.x$name %in% SKIP_R_BINDIND) %>%
#   map_chr(function(m) {
#     glue::glue("{cpp_return_type(m)} {cpp_function_name(m)} ({cpp_signature(m)}) {{ {cpp_code(m)} }}")
#   })


# simple_methods() %>%
#
#   purrr::map(~.x$arguments %>% purrr::map_chr(~.x$dynamic_type)) %>%
#   unlist() %>%
#   unique()
#
# simple_methods() %>%
#   purrr::map(~.x$method_of) %>%
#   unlist() %>%
#   unique()
#
#
