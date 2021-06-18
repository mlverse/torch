#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void* _lantern_ScriptModule_parameters (void* module)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_parameters();
    return (void*) new torch::jit::named_parameter_list(output);
}

void* _lantern_ScriptModule_forward (void* module, void* inputs)
{
    Stack inputs_ = reinterpret_cast<LanternObject<Stack>*>(inputs)->get();
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);

    auto outputs = new LanternObject<torch::jit::Stack>();
    auto out = module_->forward(inputs_);
    outputs->get().push_back(out);  

    return (void*) outputs;
}

void _lantern_ScriptModule_train (void* module, bool on)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    module_->train(on);
}

void _lantern_ScriptModule_to (void* module, void* device, bool non_blocking)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto device_ = reinterpret_cast<LanternPtr<torch::Device>*>(device);
    module_->to(device_->get(), non_blocking);
}

void _lantern_ScriptModule_set_optimized (void* module, bool o)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    module_->set_optimized(o);
}

bool _lantern_ScriptModule_is_training (void* module)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    return module_->is_training();
}

bool _lantern_ScriptModule_is_optimized (void* module)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    return module_->is_optimized();
}

void* _lantern_ScriptModule_modules (void* module, bool o)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_modules();
    return (void*) new torch::jit::named_module_list(output);
}

void _lantern_ScriptModule_register_parameter (void* module, void* name, void* v, bool is_buffer)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto v_ = reinterpret_cast<LanternObject<torch::Tensor>*>(v)->get();
    module_->register_parameter(name_, v_, is_buffer);
}

void _lantern_ScriptModule_register_buffer (void* module, void* name, void* v)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto v_ = reinterpret_cast<LanternObject<torch::Tensor>*>(v)->get();
    module_->register_buffer(name_, v_);
}

void _lantern_ScriptModule_register_module (void* self, void* name, void* module)
{
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    self_->register_module(name_, *module_);
}

void _lantern_ScriptModule_register_attribute (void* module, void* name, void* t, void* v, bool is_param, bool is_buffer)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();

    auto t_ = reinterpret_cast<c10::TypePtr*>(t);
    auto v_ = reinterpret_cast<c10::IValue*>(v);

    module_->register_attribute(name_, *t_, *v_, is_param, is_buffer);
}

