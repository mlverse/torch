#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void* _lantern_jit_compile(void* source, void* cu) {
  LANTERN_FUNCTION_START
  const auto source_ = from_raw::string(source);
  auto result = std::move(*torch::jit::compile(source_).get());
  return make_raw::CompilationUnit(result);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_compile_list_methods(void* cu) {
  LANTERN_FUNCTION_START
  auto funs = from_raw::CompilationUnit(cu).get_functions();
  std::vector<std::string> names;
  for (const auto& f : funs) {
    names.push_back(f->name());
  }
  return make_raw::vector::string(names);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_compile_get_method(void* cu, void* name) {
  LANTERN_FUNCTION_START
  auto name_ = from_raw::string(name);
  return (void*)from_raw::CompilationUnit(cu).find_function(name_);
  LANTERN_FUNCTION_END
}

void * _lantern_jit_get_all_operators_names () {
  LANTERN_FUNCTION_START
  auto ops = torch::jit::getAllOperators();
  std::vector<std::string> names;
  for (const auto& op : ops) {
    names.push_back(op->schema().name());
  }
  return make_raw::vector::string(names);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_get_operation_schema (void* name) {
  LANTERN_FUNCTION_START
  auto name_ = from_raw::string(name);
  auto op_name = c10::Symbol::fromQualString(name_);
  auto op = torch::jit::getAllOperatorsFor(op_name);
  return make_raw::FunctionSchema(op[0]->schema());
  LANTERN_FUNCTION_END
}

void* _lantern_jit_FunctionSchema_name (void* schema) {
  auto schema_ = from_raw::FunctionSchema(schema);
  return make_raw::string(schema_.name());
}

// https://cs.github.com/pytorch/pytorch/blob/47834679ba2f869e66450a74e2add4c04f0006e9/torch/csrc/jit/python/pybind_utils.h#L874
// https://cs.github.com/pytorch/pytorch/blob/47834679ba2f869e66450a74e2add4c04f0006e9/torch/csrc/jit/python/pybind_utils.h#L1137