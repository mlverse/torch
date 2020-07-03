#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"
extern std::string *pLanternLastError;

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

void lantern_Tensor_set_grad_(void* self, void * new_grad)
{
  auto t =  reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  auto g = reinterpret_cast<LanternObject<torch::Tensor> *>(new_grad)->get();
  t.grad() = g;
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
        if (out == (void*)NULL) {
          torch::Tensor empty;
          return empty;
        }
        auto ten = reinterpret_cast<LanternObject<torch::Tensor> *>(out)->get();
        return ten;
    };
    return (void *)new LanternObject<std::function<torch::Tensor(torch::Tensor)>>(out);
}

void lantern_Tensor_remove_hook(void *self, unsigned int pos)
{
    reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get().remove_hook(pos);
}

void *lantern_variable_list_new()
{
    auto out = new LanternObject<torch::autograd::variable_list>();
    return (void *)out;
}

void lantern_variable_list_push_back(void *self, void *x)
{
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(self)->get().push_back(t);
}

void *lantern_variable_list_get(void *self, int64_t i)
{
    auto s = reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(self)->get();
    torch::Tensor out = s[i];
    return (void *)new LanternObject<torch::Tensor>(out);
}

int64_t lantern_variable_list_size(void *self)
{
    auto s = reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(self)->get();
    return s.size();
}

void *lantern_Function_lambda(void *(*fun)(void *, void *, void *), void *custom)
{
    auto out = [fun, custom](torch::autograd::LanternAutogradContext *ctx, torch::autograd::variable_list inputs) {
        auto out = (*fun)(custom, (void *)ctx, (void *)new LanternObject<torch::autograd::variable_list>(inputs));
        auto vl = reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(out)->get();
        return vl;
    };
    return (void *)new LanternObject<std::function<torch::autograd::variable_list(torch::autograd::LanternAutogradContext *, torch::autograd::variable_list)>>(out);
}

void *lantern_Function_apply(void *inputs, void *forward, void *backward)
{
    LANTERN_FUNCTION_START
    auto out = torch::autograd::LanternFunction::apply(
        reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(inputs)->get(),
        reinterpret_cast<LanternObject<std::function<torch::autograd::variable_list(torch::autograd::LanternAutogradContext *, torch::autograd::variable_list)>> *>(forward)->get(),
        reinterpret_cast<LanternObject<std::function<torch::autograd::variable_list(torch::autograd::LanternAutogradContext *, torch::autograd::variable_list)>> *>(backward)->get());

    return (void *)new LanternObject<torch::autograd::variable_list>(out);
    LANTERN_FUNCTION_END
}

void lantern_AutogradContext_save_for_backward(void *self, void *vars)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    ctx->save_for_backward(reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(vars)->get());
}

void *lantern_AutogradContext_get_saved_variables(void *self)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    return (void *)new LanternObject<torch::autograd::variable_list>(ctx->get_saved_variables());
}

void lantern_AutogradContext_set_arguments(void *self, void *names, void *needs_grad)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    ctx->set_arguments(
        *reinterpret_cast<std::vector<std::string> *>(names),
        *reinterpret_cast<std::vector<bool> *>(needs_grad));
}

void *lantern_AutogradContext_get_argument_names(void *self)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    return (void *)new std::vector<std::string>(ctx->get_argument_names());
}

void *lantern_AutogradContext_get_argument_needs_grad(void *self)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    return (void *)new std::vector<bool>(ctx->get_argument_needs_grad());
}

void lantern_AutogradContext_set_saved_variables_names(void *self, void *names)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    ctx->set_saved_variables_names(*reinterpret_cast<std::vector<std::string> *>(names));
}

void *lantern_AutogradContext_get_saved_variables_names(void *self)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    return (void *)new std::vector<std::string>(ctx->get_saved_variables_names());
}

void lantern_AutogradContext_mark_dirty(void *self, void *inputs)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    auto vars = reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(inputs)->get();
    ctx->mark_dirty(vars);
}

void lantern_AutogradContext_mark_non_differentiable(void *self, void *outputs)
{
    auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
    auto vars = reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(outputs)->get();
    ctx->mark_non_differentiable(vars);
}

void lantern_autograd_backward(void *tensors, void *grad_tensors, bool retain_graph, bool create_graph)
{
    torch::autograd::backward(
        ((LanternObject<torch::autograd::variable_list> *)tensors)->get(),
        ((LanternObject<torch::autograd::variable_list> *)grad_tensors)->get(),
        retain_graph, create_graph);
}

void *lantern_autograd_grad(void *outputs, void *inputs, void *grad_outputs,
                            bool retain_graph, bool create_graph, bool allow_unused)
{
    auto out = torch::autograd::grad(
        reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(outputs)->get(),
        reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(inputs)->get(),
        reinterpret_cast<LanternObject<torch::autograd::variable_list> *>(grad_outputs)->get(),
        retain_graph,
        create_graph,
        allow_unused);
    return (void *)new LanternObject<torch::autograd::variable_list>(out);
}

void *lantern_Tensor_grad_fn(void *self)
{
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
    auto f = t.grad_fn().get();
    return reinterpret_cast<void *>(f);
}

const char *lantern_Node_name(void *self)
{
    auto x = new std::string(reinterpret_cast<torch::autograd::Node *>(self)->name());
    return x->c_str();
}

void *lantern_Node_next_edges(void *self)
{
    auto n = reinterpret_cast<torch::autograd::Node *>(self);
    auto out = new LanternObject<torch::autograd::edge_list>(n->next_edges());
    return (void *)out;
}

int64_t lantern_edge_list_size(void *self)
{
    auto e = reinterpret_cast<LanternObject<torch::autograd::edge_list> *>(self)->get();
    return e.size();
}

void *lantern_edge_list_at(void *self, int64_t i)
{
    auto e = reinterpret_cast<LanternObject<torch::autograd::edge_list> *>(self)->get();
    auto out = new LanternObject<torch::autograd::Edge>(e.at(i));
    return (void *)out;
}

void *lantern_Edge_function(void *self)
{
    auto e = reinterpret_cast<LanternObject<torch::autograd::Edge> *>(self)->get();
    auto out = e.function.get();
    return (void *)out;
}

void test_grad_fn()
{
    auto x = torch::randn({1}, torch::requires_grad());
    auto y = 2 * x;
    std::cout << "take 1" << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;
    std::cout << "take 2" << std::endl;
    auto o = (void *)new LanternObject<torch::Tensor>(y);
    auto s = std::string(lantern_Node_name(lantern_Tensor_grad_fn(o)));
    std::cout << s << std::endl;

    std::cout << "take 3" << std::endl;
    auto gf = lantern_Tensor_grad_fn(o);
    auto el = lantern_Node_next_edges(gf);
    std::cout << "hi" << std::endl;
    std::cout << lantern_edge_list_size(el) << std::endl;
    auto e = lantern_edge_list_at(el, 0);
    auto f = lantern_Edge_function(e);
    std::cout << "hey" << std::endl;
    auto k = std::string(lantern_Node_name(f));
    std::cout << k << std::endl;
}