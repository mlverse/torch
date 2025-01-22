#include <torch/csrc/jit/serialization/import.h>
#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void* _lantern_ScriptModule_new(void* cu, void* name) {
  // this pointer shouldn't use the default deleter as its memory is managed in
  // the R side.
  auto cu_ = std::shared_ptr<torch::CompilationUnit>(
      &from_raw::CompilationUnit(cu), [](void* x) {});

  auto name_ = *reinterpret_cast<std::string*>(name);
  auto module = new torch::jit::Module(name_, cu_);

  // for trace jitted models, we add a custom forward method that respects
  // the train/eval mode
  module->register_attribute("training", torch::BoolType::get(), true);
  module->define(R"(
  def train(self, x: Optional[List[bool]] = None) -> List[bool]:
      if x is None:
          x = [True]  # Default value
      self.training = x[0]
      return x
  )");

  module->define(R"(
    def eval(self) -> List[bool]:
      self.training = False
      return [False]
  )");

  module->define(R"(
    def is_training(self) -> List[bool]:
      return [self.training]
  )");

  return (void*) module;
}

void* _lantern_ScriptModule_parameters(void* module, bool recurse) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto output = module_->named_parameters(recurse);
  return (void*)new torch::jit::named_parameter_list(output);
}

void* _lantern_ScriptModule_buffers(void* module, bool recurse) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto output = module_->named_buffers(recurse);
  return (void*)new torch::jit::named_buffer_list(output);
}

void* _lantern_ScriptModule_forward(void* module, void* inputs) {
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);

  auto outputs = torch::jit::Stack();
  auto out = module_->forward(inputs_);
  outputs.push_back(out);

  return make_ptr<torch::jit::Stack>(outputs);
}

void _lantern_ScriptModule_train(void* module, bool on) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  module_->train(on);
}

void _lantern_ScriptModule_to(void* module, void* device, bool non_blocking) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto device_ = from_raw::Device(device);
  module_->to(device_, non_blocking);
}

void _lantern_ScriptModule_set_optimized(void* module, bool o) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  module_->set_optimized(o);
}

bool _lantern_ScriptModule_is_training(void* module) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  return module_->is_training();
}

bool _lantern_ScriptModule_is_optimized(void* module) {
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  return module_->is_optimized();
}

void* _lantern_ScriptModule_modules(void* module) {
  LANTERN_FUNCTION_START
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto output = module_->named_modules();
  return (void*)new torch::jit::named_module_list(output);
  LANTERN_FUNCTION_END
}

void* _lantern_ScriptModule_children(void* module) {
  LANTERN_FUNCTION_START
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto output = module_->named_children();
  return (void*)new torch::jit::named_module_list(output);
  LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_register_parameter(void* module, void* name, void* v,
                                              bool is_buffer) {
  LANTERN_FUNCTION_START
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto name_ = from_raw::string(name);
  auto v_ = from_raw::Tensor(v);
  module_->register_parameter(name_, v_, is_buffer);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_buffer(void* module, void* name, void* v) {
  LANTERN_FUNCTION_START
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto name_ = from_raw::string(name);
  auto v_ = from_raw::Tensor(v);
  module_->register_buffer(name_, v_);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_module(void* self, void* name,
                                           void* module) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto name_ = from_raw::string(name);
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  self_->register_module(name_, *module_);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_attribute(void* module, void* name, void* t,
                                              void* v, bool is_param,
                                              bool is_buffer) {
  LANTERN_FUNCTION_START
  auto module_ = reinterpret_cast<torch::jit::script::Module*>(module);
  auto name_ = from_raw::string(name);

  auto t_ = reinterpret_cast<c10::TypePtr*>(t);
  auto v_ = reinterpret_cast<c10::IValue*>(v);

  module_->register_attribute(name_, *t_, *v_, is_param, is_buffer);
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_ScriptModule_find_method(void* self, void* basename) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto basename_ = from_raw::string(basename);
  auto method = self_->find_method(basename_);

  if (!method.has_value()) {
    return nullptr;
  } else {
    return (void*)new torch::jit::script::Method(method.value());
  }
  LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_add_method(void* self, void* method) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto method_ = reinterpret_cast<torch::jit::Function*>(method);
  self_->type()->addMethod(method_);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_add_forward(void* self, bool list_output) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  if (list_output) {
    self_->define(R"(
      def forward(self, x) -> List[Tensor]:
        if self.training:
          return self.trainforward(x)
        else:
          return self.evalforward(x)
    )");
  } else {
    self_->define(R"(
      def forward(self, x) -> Tensor:
        if self.training:
          return self.trainforward(x)
        else:
          return self.evalforward(x)
    )");
    
  }
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_add_constant(void* self, void* name, void* value) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto name_ = reinterpret_cast<std::string*>(name);
  auto value_ = reinterpret_cast<c10::IValue*>(value);
  self_->type()->addConstant(*name_, *value_);
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_ScriptModule_find_constant(void* self, void* name) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto name_ = reinterpret_cast<std::string*>(name);
  auto constant = self_->type()->findConstant(*name_);
  if (!constant.has_value()) {
    return nullptr;
  } else {
    return (void*)new c10::IValue(constant.value());
  }
  LANTERN_FUNCTION_END
}

void* _lantern_ScriptMethod_call(void* self, void* inputs) {
  LANTERN_FUNCTION_START
  auto self_ = *reinterpret_cast<torch::jit::script::Method*>(self);
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);

  auto outputs = torch::jit::Stack();
  auto out = self_(inputs_);
  outputs.push_back(out);

  return make_ptr<torch::jit::Stack>(outputs);
  LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_save(void* self, void* path) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto path_ = from_raw::string(path);
  self_->save(path_);
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_ScriptModule_serialize(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  std::ostringstream oss(std::ios::binary);
  self_->save(oss);
  auto str = std::string(oss.str());
  return make_raw::string(str);
  LANTERN_FUNCTION_END
}

void* _lantern_ScriptModule_unserialize(void* s) {
  LANTERN_FUNCTION_START
  auto str = from_raw::string(s);
  std::istringstream input_stream(str);
  torch::jit::script::Module module;
  module = torch::jit::load(input_stream);
  return (void*)new torch::jit::script::Module(module);
  LANTERN_FUNCTION_END
}

void* _lantern_ScriptMethod_graph_print(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Method*>(self);
  std::string str = self_->graph()->toString();
  return (void*)new std::string(str);
  LANTERN_FUNCTION_END
}

void* _lantern_last_executed_optimized_graph_print() {
  LANTERN_FUNCTION_START
  auto str = torch::jit::lastExecutedOptimizedGraph()->toString();
  return (void*)new std::string(str);
  LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_save_for_mobile(void* self, void* path) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  auto path_ = from_raw::string(path);
  self_->_save_for_mobile(path_);
  LANTERN_FUNCTION_END_VOID
}
