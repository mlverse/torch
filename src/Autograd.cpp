#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"
#include <thread>

void lantern_autograd_set_grad_mode(bool enabled)
{
    torch::autograd::GradMode::set_enabled(enabled);
}

void *lantern_Tensor_grad(void *self)
{
    auto out = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get().grad();
    return (void *)new LanternObject<torch::Tensor>(out);
}

bool lantern_Tensor_requires_grad(void *self)
{
    return reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get().requires_grad();
}

void lantern_Tensor_register_hook(void *self, void *hook)
{
    auto h = reinterpret_cast<LanternObject<std::function<torch::Tensor(torch::Tensor)>> *>(hook)->get();
    auto x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    x.register_hook(h);
}

// Creating the hook in the right format to be passed to .register_hook
// It takes a pointer a function that in turn will take a pointer to a
// torch tensor and a function to apply over it.
// fun must return a pointer To a lantern object of type tensor.
void *lantern_new_hook(void *(*fun)(void *, void *), void *custom)
{
    auto out = [fun, custom](torch::Tensor grad) {
        auto out = (*fun)((void *)new LanternObject<torch::Tensor>(grad), custom);
        auto ten = reinterpret_cast<LanternObject<torch::Tensor> *>(out)->get();
        return ten;
    };
    return (void *)new LanternObject<std::function<torch::Tensor(torch::Tensor)>>(out);
}