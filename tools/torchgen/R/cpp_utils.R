
torch_argument_type_to_rcpp_argument_type <- function(dynamic_type) {

  switch(dynamic_type,
    "Tensor"             = "Rcpp::XPtr<torch::Tensor>",
    "bool"               = "bool",
    "DimnameList"        = "Rcpp::XPtr<std::vector<torch::Dimname>>",
    "TensorList"         = "Rcpp::XPtr<std::vector<torch::Tensor>>",
    "IntArrayRef"        = "std::vector<int64_t>",
    "int64_t"            = "int64_t",
    "double"             = "double",
    "std::array<bool,4>" = "std::array<bool,4>",
    "std::array<bool,3>" = "std::array<bool,3>",
    "std::array<bool,2>" = "std::array<bool,2>",
    "TensorOptions"      = "Rcpp::XPtr<torch::TensorOptions>",
    "Generator *"        = "Rcpp::XPtr<Generator *>",
    "ScalarType"         = "Rcpp::XPtr<torch::Dtype>",
    "Scalar"             = "Rcpp::XPtr<torch::Scalar>",
    "MemoryFormat"       = "Rcpp::XPtr<torch::MemoryFormat>",
    "std::string"        = "std::string"
  )

}

torch_return_type_to_rcpp_return_type <- function(dynamic_types) {

  if (length(dynamic_types) > 1)
    return("Rcpp::List")

  switch(dynamic_types,
    "Tensor"     = "Rcpp:XPtr<torch::Tensor>",
    "void"       = "void",
    "bool"       = "bool",
    "int64_t"    = "int64_t",
    "TensorList" = "Rcpp::XPtr<torch::TensorList>",
    "double"     = "double",
    "QScheme"    = "Rcpp::XPtr<torch::QScheme>",
    "Scalar"     = "Rcpp::XPtr<torch::Scalar>",
    "ScalarType" = "Rcpp::XPtr<torch::Dtype>"
  )

}

torch_object_from_rcpp_object <- function(dynamic_type, name) {

  ptr <- function(name) {
    glue::glue("* {name}")
  }

  switch(dynamic_type,
    "Tensor"             = ptr(name),
    "bool"               = name,
    "DimnameList"        = ptr(name),
    "TensorList"         = ptr(name),
    "IntArrayRef"        = name,
    "int64_t"            = name,
    "double"             = name,
    "std::array<bool,4>" = name,
    "std::array<bool,3>" = name,
    "std::array<bool,2>" = name,
    "TensorOptions"      = ptr(name),
    "Generator *"        = ptr(name),
    "ScalarType"         = ptr(name),
    "Scalar"             = ptr(name),
    "MemoryFormat"       = ptr(name),
    "std::string"        = name
  )

}
