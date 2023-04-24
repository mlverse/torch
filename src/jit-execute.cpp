#include <torch.h>

// [[Rcpp::export]]
torch::vector::string cpp_jit_all_operators() {
  return lantern_jit_all_operators();
}

// [[Rcpp::export]]
torch::jit::FunctionSchema cpp_jit_operator_info(torch::string name) {
  return lantern_jit_operator_info(name.get());
}

// [[Rcpp::export]]
torch::jit::FunctionSchemaList cpp_jit_all_schemas_for(torch::string name) {
  return lantern_jit_all_schemas_for(name.get());
}

// [[Rcpp::export]]
torch::jit::Stack cpp_jit_execute(torch::string name, torch::jit::Stack stack) {
  return lantern_jit_execute(name.get(), stack.get());
}

// [[Rcpp::export]]
torch::string function_schema_name(XPtrTorchFunctionSchema schema) {
  return lantern_function_schema_name(schema.get());
}

// [[Rcpp::export]]
torch::jit::ArgumentList function_schema_arguments(XPtrTorchFunctionSchema schema) {
  return lantern_function_schema_arguments(schema.get());
}

// [[Rcpp::export]]
torch::string function_schema_argument_name(XPtrTorchFunctionSchemaArgument arg) {
  return lantern_function_schema_argument_name(arg.get());
}

// [[Rcpp::export]]
torch::string function_schema_argument_type(XPtrTorchFunctionSchemaArgument arg) {
  return lantern_function_schema_argument_type(arg.get());
}

// returns are of type Argument, as well
// [[Rcpp::export]]
torch::jit::ArgumentList function_schema_returns(XPtrTorchFunctionSchema schema) {
  return lantern_function_schema_returns(schema.get());
}

// [[Rcpp::export]]
torch::string function_schema_return_type(XPtrTorchFunctionSchemaArgument ret) {
  return lantern_function_schema_return_type(ret.get());
}

