#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"
#include <iterator>

using namespace torch::jit::tracer;
using namespace torch::jit;

int _lantern_jit_named_parameter_list_size (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::named_parameter_list *>(self);
    return self_->size();
}

void* _lantern_jit_named_parameter_list_tensors (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::named_parameter_list *>(self);
    auto outputs = new LanternObject<std::vector<torch::Tensor>>();
    for (auto el : *self_)
    {
        outputs->get().push_back(el.value);
    }
    return (void*) outputs;
}

void* _lantern_jit_named_parameter_list_names (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::named_parameter_list *>(self);
    auto outputs = new std::vector<std::string>();
    for (auto el : *self_)
    {
        outputs->push_back(el.name);
    }
    return (void*) outputs;
}