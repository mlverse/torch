#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

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
    auto h = reinterpret_cast<LanternObject<std::function<void(torch::Tensor)>> *>(hook)->get();
    auto x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    x.register_hook(h);
}

// Creating the hook in the right format to be passed to .register_hook
// It takes a pointer a function that in turn will take a pointer to a
// torch tensor and a function to apply over it.
void *lantern_new_hook(void (*fun)(void *, void *), void *custom)
{
    auto out = [fun, custom](torch::Tensor grad) {
        (*fun)((void *)new LanternObject<torch::Tensor>(grad), custom);
    };
    return (void *)new LanternObject<std::function<void(torch::Tensor)>>(out);
}

// Examples of usage of register hook
// should be excluded before the PR is merged.
void fun(void *x, void *custom)
{
    (*reinterpret_cast<std::function<void(void *)> *>(custom))(x);
}

void lantern_test_register_hook()
{
    auto x = torch::randn(1, torch::requires_grad());
    auto y = (void *)new torch::Tensor(x);
    auto custom = (void *)new std::function<void(void *)>([](void *x) { std::cout << "hello hello" << std::endl; });
    auto f = lantern_new_hook(&fun, custom);
    lantern_Tensor_register_hook(y, f);
    auto z = reinterpret_cast<LanternObject<torch::Tensor> *>(y)->get();
    auto k = 2 * z;
    k.backward();
    std::cout << z.grad() << std::endl;
}
