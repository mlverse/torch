#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void* _lantern_CompilationUnit_new() {
  LANTERN_FUNCTION_START;
  auto cu = torch::jit::CompilationUnit();
  return make_raw::CompilationUnit(cu);
  LANTERN_FUNCTION_END;
}

void* _lantern_create_traceable_fun(void* (*r_caller)(void*, void*), void* fn) {
  LANTERN_FUNCTION_START;
  std::function<Stack(Stack)> tr_fn = [r_caller, fn](Stack x) {
    // auto r_fn = *reinterpret_cast<std::function<void*(void*)>*>(fn);
    auto tmp = Stack(x);
    void* out = (*r_caller)((void*)(&tmp), fn);
    // if the R function call fails itt will return nullptr by convention.
    if (out == nullptr)
      throw std::runtime_error("Error in the R function execution.");

    return *reinterpret_cast<Stack*>(out);
  };

  return (void*)new std::function<Stack(Stack)>(tr_fn);
  LANTERN_FUNCTION_END;
}

void* _lantern_trace_fn(void* fn, void* inputs, void* compilation_unit,
                        bool strict, void* module, void* name,
                        bool should_mangle) {
  LANTERN_FUNCTION_START;
  std::function<Stack(Stack)> fn_ =
      *reinterpret_cast<std::function<Stack(Stack)>*>(fn);
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto name_ = from_raw::string(name);

  std::function<std::string(const torch::autograd::Variable&)> var_fn =
      [](const torch::autograd::Variable& x) { return ""; };

  auto traced =
      torch::jit::tracer::trace(inputs_, fn_, var_fn, strict, false, module_);

  auto tr_fn =
      from_raw::CompilationUnit(compilation_unit)
          .create_function(name_, std::get<0>(traced)->graph, should_mangle);

  return (void*)tr_fn;
  LANTERN_FUNCTION_END;
}

void* _lantern_call_traced_fn(void* fn, void* inputs) {
  LANTERN_FUNCTION_START
  Function* fn_ = reinterpret_cast<Function*>(fn);
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);

  auto outputs = torch::jit::Stack();
  auto out = (*fn_)(inputs_);
  outputs.push_back(out);

  return make_ptr<torch::jit::Stack>(outputs);
  LANTERN_FUNCTION_END
}

void addFunctionToModule(Module& module, const Function* func) {
  LANTERN_FUNCTION_START
  // Make a graph with a fake self argument
  auto graph = func->graph()->copy();
  auto v = graph->insertInput(0, "self");
  v->setType(module._ivalue()->type());
  const auto name = QualifiedName(*module.type()->name(), "forward");
  auto method =
      module._ivalue()->compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_traced_fn_save(void* fn, const char* filename) {
  LANTERN_FUNCTION_START;
  Function* fn_ = reinterpret_cast<Function*>(fn);
  auto filename_ = std::string(filename);

  Module module("__torch__.PlaceholderModule");

  module.register_attribute("training", BoolType::get(), true);
  addFunctionToModule(module, fn_);
  module.save(filename_);
  LANTERN_FUNCTION_END_VOID;
}

const char* _lantern_traced_fn_graph_print(void* fn) {
  LANTERN_FUNCTION_START
  Function* fn_ = reinterpret_cast<Function*>(fn);
  std::string str = fn_->graph()->toString();
  char* cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void* _lantern_jit_load(const char* path) {
  LANTERN_FUNCTION_START
  auto module = new torch::jit::script::Module();
  *module = torch::jit::load(std::string(path));
  return (void*)module;
  LANTERN_FUNCTION_END
}

void* _lantern_call_jit_script(void* module, void* inputs) {
  LANTERN_FUNCTION_START
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);

  auto outputs = torch::jit::Stack();
  auto out = module_->forward(inputs_);
  outputs.push_back(out);

  return make_ptr<torch::jit::Stack>(outputs);
  LANTERN_FUNCTION_END
}

void _trace_r_nn_module() {
  LANTERN_FUNCTION_START;
  auto x = torch::randn({10, 10});

  torch::jit::Stack inputs;
  inputs.push_back(x);

  std::function<torch::jit::Stack(torch::jit::Stack)> fn =
      [](torch::jit::Stack x) {
        auto res = torch::relu(x[0].toTensor());
        res = torch::exp(res);
        torch::jit::Stack output;
        output.push_back(res);
        return output;
      };

  std::function<std::string(const torch::autograd::Variable&)> var_fn =
      [](const torch::autograd::Variable& x) { return ""; };

  auto traced = torch::jit::tracer::trace(inputs, fn, var_fn);

  auto cu = torch::jit::CompilationUnit();
  auto z = cu.create_function("name", std::get<0>(traced)->graph, true);

  std::cout << "calling JIT" << std::endl;

  auto t = std::get<0>(traced);
  std::cout << t->graph->toString() << std::endl;
  auto node = t->graph->param_node();
  std::cout << node->attributeNames() << std::endl;

  std::cout << "calling JIT" << std::endl;
  std::cout << (*z)(inputs).toTensor() << std::endl;
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_traced_fn_save_for_mobile(void* fn, const char* filename) {
  LANTERN_FUNCTION_START;
  Function* fn_ = reinterpret_cast<Function*>(fn);
  auto filename_ = std::string(filename);

  Module module("__torch__.PlaceholderModule");

  module.register_attribute("training", BoolType::get(), true);
  addFunctionToModule(module, fn_);
  module._save_for_mobile(filename_);
  LANTERN_FUNCTION_END_VOID;
}
