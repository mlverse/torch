#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void* _lantern_ScriptModule_new (void* cu, void* name)
{
    // this pointer shouldn't use the default deleter as its memory is managed in 
    // the R side.
    auto cu_ = std::shared_ptr<torch::CompilationUnit>(
        reinterpret_cast<torch::CompilationUnit*>(cu), 
        [](void* x){}
        );
    auto name_ = *reinterpret_cast<std::string*>(name);
    return (void*) new torch::jit::Module(name_, cu_);
}

void* _lantern_ScriptModule_parameters (void* module, bool recurse)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_parameters(recurse);
    return (void*) new torch::jit::named_parameter_list(output);
}

void* _lantern_ScriptModule_buffers (void* module, bool recurse)
{
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_buffers(recurse);
    return (void*) new torch::jit::named_buffer_list(output);
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

void* _lantern_ScriptModule_modules (void* module)
{
    LANTERN_FUNCTION_START
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_modules();
    return (void*) new torch::jit::named_module_list(output);
    LANTERN_FUNCTION_END
}

void* _lantern_ScriptModule_children (void* module)
{
    LANTERN_FUNCTION_START
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto output = module_->named_children();
    return (void*) new torch::jit::named_module_list(output);
    LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_register_parameter (void* module, void* name, void* v, bool is_buffer)
{
    LANTERN_FUNCTION_START
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto v_ = reinterpret_cast<LanternObject<torch::Tensor>*>(v)->get();
    module_->register_parameter(name_, v_, is_buffer);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_buffer (void* module, void* name, void* v)
{
    LANTERN_FUNCTION_START
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto v_ = reinterpret_cast<LanternObject<torch::Tensor>*>(v)->get();
    module_->register_buffer(name_, v_);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_module (void* self, void* name, void* module)
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    self_->register_module(name_, *module_);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_register_attribute (void* module, void* name, void* t, void* v, bool is_param, bool is_buffer)
{
    LANTERN_FUNCTION_START
    auto module_ = reinterpret_cast<torch::jit::script::Module *>(module);
    auto name_ = reinterpret_cast<LanternObject<std::string>*>(name)->get();

    auto t_ = reinterpret_cast<c10::TypePtr*>(t);
    auto v_ = reinterpret_cast<c10::IValue*>(v);

    module_->register_attribute(name_, *t_, *v_, is_param, is_buffer);
    LANTERN_FUNCTION_END_VOID
}

void* _lantern_ScriptModule_find_method (void* self, void* basename)
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto basename_ = reinterpret_cast<LanternObject<std::string>*>(basename)->get();
    auto method = self_->find_method(basename_);

    if (!method.has_value())
    {
        return nullptr;
    }
    else
    {
        return (void*) new torch::jit::script::Method(method.value());
    }
    LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_add_method (void* self, void* method) 
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto method_ = reinterpret_cast<torch::jit::Function *>(method);
    self_->type()->addMethod(method_);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_ScriptModule_add_constant (void* self, void* name, void* value)
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto name_ = reinterpret_cast<std::string*>(name);
    auto value_ = reinterpret_cast<c10::IValue*>(value);
    self_->type()->addConstant(*name_, *value_);
    LANTERN_FUNCTION_END_VOID
}

void* _lantern_ScriptModule_find_constant (void* self, void* name)
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto name_ = reinterpret_cast<std::string*>(name);
    auto constant = self_->type()->findConstant(*name_);
    if (!constant.has_value())
    {
        return nullptr;
    }
    else
    {
        return (void*) new c10::IValue(constant.value());
    }
    LANTERN_FUNCTION_END
}

void* _lantern_ScriptMethod_call (void* self, void* inputs)
{
    LANTERN_FUNCTION_START
    auto self_ = *reinterpret_cast<torch::jit::script::Method *>(self);
    Stack inputs_ = reinterpret_cast<LanternObject<Stack>*>(inputs)->get();
    
    auto outputs = new LanternObject<torch::jit::Stack>();
    auto out = self_(inputs_);
    outputs->get().push_back(out);  

    return (void*) outputs;
    LANTERN_FUNCTION_END
}

void _lantern_ScriptModule_save (void* self, void* path)
{
    LANTERN_FUNCTION_START
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    auto path_ = reinterpret_cast<LanternObject<std::string>*>(path)->get();
    self_->save(path_);
    LANTERN_FUNCTION_END_VOID
}