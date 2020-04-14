#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"
#include "Function.h"
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

unsigned int lantern_Tensor_register_hook(void *self, void *hook)
{
    auto h = reinterpret_cast<LanternObject<std::function<torch::Tensor(torch::Tensor)>> *>(hook)->get();
    auto x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    return x.register_hook(h);
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

void lantern_Tensor_remove_hook(void *self, unsigned int pos)
{
    reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get().remove_hook(pos);
}

struct MyFunction : public LanternFunction
{
    variable_list forward(LanternAutogradContext *ctx, variable_list args)
    {
        ctx->save_for_backward(args);
        return {args[0] + args[1] + args[0] * args[1]};
    }

    static variable_list backward(LanternAutogradContext *ctx, variable_list grad_output)
    {
        auto saved = ctx->get_saved_variables();
        auto var1 = saved[0];
        auto var2 = saved[1];
        variable_list output = {grad_output[0] + grad_output[0] * var2, Variable(), grad_output[0] + grad_output[0] * var1};
        return output;
    }
};

void test_custom_function()
{

    int mul = 2;
    Variable x = torch::randn({5, 5}, torch::requires_grad());
    Variable y = torch::randn({5, 5}, torch::requires_grad());

    auto res = LanternFunction::apply(
        {x, y},
        [&](LanternAutogradContext *ctx, variable_list args) {
            ctx->save_for_backward(args);
            ctx->saved_data["mul"] = mul;
            return variable_list({args[0] + mul * args[1] + args[0] * args[1]});
        },
        [](LanternAutogradContext *ctx, variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            int mul = ctx->saved_data["mul"].toInt();
            auto var1 = saved[0];
            auto var2 = saved[1];
            variable_list output = {grad_output[0] + grad_output[0] * var2,
                                    grad_output[0] * mul + grad_output[0] * var1};
            return output;
        });

    auto go = torch::ones({}, torch::requires_grad());
    res[0].sum().backward(go, false, true);

    std::cout << x.grad() << std::endl;
    std::cout << y.grad() << std::endl;
}
