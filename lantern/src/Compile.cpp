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