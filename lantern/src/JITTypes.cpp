#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"
#include <iterator>

using namespace torch::jit::tracer;
using namespace torch::jit;

template<class T>
int jit_type_size (void* self)
{
   auto self_ = reinterpret_cast<T *>(self);
   return self_->size(); 
}

int _lantern_jit_named_parameter_list_size (void* self)
{
    return jit_type_size<torch::jit::named_parameter_list>(self);
}

int _lantern_jit_named_module_list_size (void* self)
{
    return jit_type_size<torch::jit::named_module_list>(self);
}

int _lantern_jit_named_buffer_list_size (void* self)
{
    return jit_type_size<torch::jit::named_buffer_list>(self);
}

template<class T>
void* jit_named_list_tensors (void* self)
{
    auto self_ = reinterpret_cast<T *>(self);
    auto outputs = new LanternObject<std::vector<torch::Tensor>>();
    for (auto el : *self_)
    {
        outputs->get().push_back(el.value);
    }
    return (void*) outputs;
}

void* _lantern_jit_named_parameter_list_tensors (void* self)
{
    return jit_named_list_tensors<torch::jit::named_parameter_list>(self);
}

void* _lantern_jit_named_buffer_list_tensors (void* self)
{
    return jit_named_list_tensors<torch::jit::named_buffer_list>(self);
}

void* _lantern_jit_named_module_list_module_at (void* self, int64_t index)
{
    auto self_ = reinterpret_cast<torch::jit::named_module_list *>(self);
    
    int i = 0;
    torch::jit::Module out;
    for (auto el : *self_) {
        if (i == index)
        {
            out = el.value;
            break;
        }
        i++;    
    }

    return (void*) new torch::jit::Module(out);
}

template <class T>
void* jit_type_names (void* self)
{
    auto self_ = reinterpret_cast<T *>(self);
    auto outputs = new std::vector<std::string>();
    for (auto el : *self_)
    {
        outputs->push_back(el.name);
    }
    return (void*) outputs;
}

void* _lantern_jit_named_parameter_list_names (void* self)
{
    return jit_type_names<torch::jit::named_parameter_list>(self);
}

void* _lantern_jit_named_module_list_names (void* self)
{
    return jit_type_names<torch::jit::named_module_list>(self);
}

void* _lantern_jit_named_buffer_list_names (void* self)
{
    return jit_type_names<torch::jit::named_buffer_list>(self);
}