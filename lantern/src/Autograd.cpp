#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include <thread>

#include "Autograd.h"
#include "Function.h"
#include "lantern/lantern.h"
#include "utils.hpp"

void _lantern_autograd_set_grad_mode(bool enabled) {
  LANTERN_FUNCTION_START
  torch::autograd::GradMode::set_enabled(enabled);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_autograd_set_detect_anomaly(bool enabled) {
  LANTERN_FUNCTION_START
  torch::autograd::AnomalyMode::set_enabled(enabled);
  LANTERN_FUNCTION_END_VOID
}

bool _lantern_autograd_detect_anomaly_is_enabled() {
  LANTERN_FUNCTION_START
  return torch::autograd::AnomalyMode::is_enabled();
  LANTERN_FUNCTION_END
}

bool _lantern_autograd_is_enabled() {
  LANTERN_FUNCTION_START
  return torch::autograd::GradMode::is_enabled();
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_grad(void *self) {
  LANTERN_FUNCTION_START
  auto out = from_raw::Tensor(self).grad();
  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}

void _lantern_Tensor_set_grad_(void *self, void *new_grad) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(self);
  auto g = from_raw::Tensor(new_grad);
  t.mutable_grad() = g;
  LANTERN_FUNCTION_END_VOID
}

bool _lantern_Tensor_requires_grad(void *self) {
  LANTERN_FUNCTION_START
  return from_raw::Tensor(self).requires_grad();
  LANTERN_FUNCTION_END
}

unsigned int _lantern_Tensor_register_hook(void *self, void *hook) {
  LANTERN_FUNCTION_START
  auto h =
      *reinterpret_cast<std::function<torch::Tensor(torch::Tensor)> *>(hook);
  auto x = from_raw::Tensor(self);
  return x.register_hook(h);
  LANTERN_FUNCTION_END_RET(0)
}

// Creating the hook in the right format to be passed to .register_hook
// It takes a pointer a function that in turn will take a pointer to a
// torch tensor and a function to apply over it.
// fun must return a pointer To a lantern object of type tensor.
void *_lantern_new_hook(void *(*fun)(void *, void *), void *custom) {
  LANTERN_FUNCTION_START
  auto out = [fun, custom](torch::Tensor grad) {
    auto out = (*fun)(make_raw::Tensor(grad), custom);
    if (out == (void *)NULL) {
      torch::Tensor empty;
      return empty;
    }
    auto ten = from_raw::Tensor(out);
    return ten;
  };
  return (void *)new std::function<torch::Tensor(torch::Tensor)>(out);
  LANTERN_FUNCTION_END
}

void _lantern_Tensor_remove_hook(void *self, unsigned int pos) {
  LANTERN_FUNCTION_START
  from_raw::Tensor(self).remove_hook(pos);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_variable_list_new() {
  LANTERN_FUNCTION_START
  return make_raw::variable_list({});
  LANTERN_FUNCTION_END
}

void _lantern_variable_list_push_back(void *self, void *x) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(x);
  from_raw::variable_list(self).push_back(t);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_variable_list_get(void *self, int64_t i) {
  LANTERN_FUNCTION_START
  auto s = from_raw::variable_list(self);
  torch::Tensor out = s[i];
  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}

int64_t _lantern_variable_list_size(void *self) {
  LANTERN_FUNCTION_START
  auto s = from_raw::variable_list(self);
  return s.size();
  LANTERN_FUNCTION_END_RET(0)
}

void (*delete_lambda_fun)(void *) = nullptr;

void _set_delete_lambda_fun(void (*fun)(void *)) { delete_lambda_fun = fun; }

LanternLambdaFunction::LanternLambdaFunction(autograd_fun fn, void *rcpp_fn) {
  this->fn_ = std::make_shared<autograd_fun>(fn);
  this->rcpp_fn = std::shared_ptr<void>(rcpp_fn, [](void *x) {
    if (delete_lambda_fun != nullptr) {
      (*delete_lambda_fun)(x);
    }
  });
}

void *_lantern_Function_lambda(void *(*fun)(void *, void *, void *),
                               void *custom, void(delete_out)(void *),
                               void *(*get_ptr)(void *)) {
  LANTERN_FUNCTION_START
  auto out = [fun, custom, delete_out, get_ptr](
                 torch::autograd::LanternAutogradContext *ctx,
                 torch::autograd::variable_list inputs) {
    auto out = (*fun)(custom, (void *)ctx, make_raw::variable_list(inputs));
    torch::autograd::variable_list res(
        from_raw::variable_list((*get_ptr)(out)));  // copy the output
    (*delete_out)(out);
    return res;
  };
  return (void *)new LanternLambdaFunction(out, custom);
  LANTERN_FUNCTION_END
}

void *_lantern_Function_apply(void *inputs, void *forward, void *backward) {
  LANTERN_FUNCTION_START
  auto out = torch::autograd::LanternFunction::apply(
      from_raw::variable_list(inputs), forward, backward);

  return make_raw::variable_list(out);
  LANTERN_FUNCTION_END
}

void _lantern_AutogradContext_save_for_backward(void *self, void *vars) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  ctx->save_for_backward(from_raw::variable_list(vars));
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_AutogradContext_get_saved_variables(void *self) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  return make_raw::variable_list(ctx->get_saved_variables());
  LANTERN_FUNCTION_END
}

void _lantern_AutogradContext_set_arguments(void *self, void *names,
                                            void *needs_grad) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  auto names_ = from_raw::vector::string(names);
  auto needs_grad_ = from_raw::vector::bool_t(needs_grad);
  ctx->set_arguments(names_, needs_grad_);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_AutogradContext_get_argument_names(void *self) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  return make_raw::vector::string(ctx->get_argument_names());
  LANTERN_FUNCTION_END
}

void *_lantern_AutogradContext_get_argument_needs_grad(void *self) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  return (void *)new std::vector<bool>(ctx->get_argument_needs_grad());
  LANTERN_FUNCTION_END
}

void _lantern_AutogradContext_set_saved_variables_names(void *self,
                                                        void *names) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  ctx->set_saved_variables_names(from_raw::vector::string(names));
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_AutogradContext_get_saved_variables_names(void *self) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  return make_raw::vector::string(ctx->get_saved_variables_names());
  LANTERN_FUNCTION_END
}

void _lantern_AutogradContext_mark_dirty(void *self, void *inputs) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  ctx->mark_dirty(from_raw::variable_list(inputs));
  LANTERN_FUNCTION_END_VOID
}

void _lantern_AutogradContext_mark_non_differentiable(void *self,
                                                      void *outputs) {
  LANTERN_FUNCTION_START
  auto ctx = reinterpret_cast<torch::autograd::LanternAutogradContext *>(self);
  ctx->mark_non_differentiable(from_raw::variable_list(outputs));
  LANTERN_FUNCTION_END_VOID
}

void _lantern_autograd_backward(void *tensors, void *grad_tensors,
                                bool retain_graph, bool create_graph) {
  LANTERN_FUNCTION_START
  torch::autograd::backward(from_raw::variable_list(tensors),
                            from_raw::variable_list(grad_tensors), retain_graph,
                            create_graph);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_autograd_grad(void *outputs, void *inputs, void *grad_outputs,
                             bool retain_graph, bool create_graph,
                             bool allow_unused) {
  LANTERN_FUNCTION_START
  auto out = torch::autograd::grad(from_raw::variable_list(outputs),
                                   from_raw::variable_list(inputs),
                                   from_raw::variable_list(grad_outputs),
                                   retain_graph, create_graph, allow_unused);
  return make_raw::variable_list(out);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_grad_fn(void *self) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(self);
  auto f = t.grad_fn().get();
  return reinterpret_cast<void *>(f);
  LANTERN_FUNCTION_END
}

const char *_lantern_Node_name(void *self) {
  // TODO: fix to return a pointer to string instead of char*
  LANTERN_FUNCTION_START
  auto str =
      std::string(reinterpret_cast<torch::autograd::Node *>(self)->name());
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void *_lantern_Node_next_edges(void *self) {
  LANTERN_FUNCTION_START
  auto n = reinterpret_cast<torch::autograd::Node *>(self);
  auto out = new torch::autograd::edge_list(n->next_edges());
  return (void *)out;
  LANTERN_FUNCTION_END
}

int64_t _lantern_edge_list_size(void *self) {
  LANTERN_FUNCTION_START
  auto e = *reinterpret_cast<torch::autograd::edge_list *>(self);
  return e.size();
  LANTERN_FUNCTION_END_RET(0)
}

void *_lantern_edge_list_at(void *self, int64_t i) {
  LANTERN_FUNCTION_START
  auto e = *reinterpret_cast<torch::autograd::edge_list *>(self);
  auto out = new torch::autograd::Edge(e.at(i));
  return (void *)out;
  LANTERN_FUNCTION_END
}

void *_lantern_Edge_function(void *self) {
  LANTERN_FUNCTION_START
  auto e = *reinterpret_cast<torch::autograd::Edge *>(self);
  auto out = e.function.get();
  return (void *)out;
  LANTERN_FUNCTION_END
}

void _test_grad_fn() {
  LANTERN_FUNCTION_START
  auto x = torch::randn({1}, torch::requires_grad());
  auto y = 2 * x;
  std::cout << "take 1" << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;
  std::cout << "take 2" << std::endl;
  auto o = make_raw::Tensor(y);
  auto s = std::string(_lantern_Node_name(_lantern_Tensor_grad_fn(o)));
  std::cout << s << std::endl;

  std::cout << "take 3" << std::endl;
  auto gf = _lantern_Tensor_grad_fn(o);
  auto el = _lantern_Node_next_edges(gf);
  std::cout << "hi" << std::endl;
  std::cout << _lantern_edge_list_size(el) << std::endl;
  auto e = _lantern_edge_list_at(el, 0);
  auto f = _lantern_Edge_function(e);
  std::cout << "hey" << std::endl;
  auto k = std::string(_lantern_Node_name(f));
  std::cout << k << std::endl;
  LANTERN_FUNCTION_END_VOID
}