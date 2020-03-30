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

void lantern_Tensor_register_hook(void *self)
{
    auto x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    x.register_hook([](torch::Tensor grad) {
        std::cout << "hello" << std::endl;
    });
}

void lantern_test_register_hook()
{
    auto x = torch::randn(1, torch::requires_grad());
    auto y = (void *)new torch::Tensor(x);
    lantern_Tensor_register_hook(y);
    auto z = reinterpret_cast<LanternObject<torch::Tensor> *>(y)->get();
    auto k = 2 * z;
    k.backward();
    std::cout << z.grad() << std::endl;
}
