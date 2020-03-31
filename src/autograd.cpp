#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
void cpp_autograd_set_grad_mode (bool enabled) {
  lantern_autograd_set_grad_mode(enabled);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_tensor_grad (Rcpp::XPtr<XPtrTorchTensor> self) {
  return make_xptr<XPtrTorchTensor>(lantern_Tensor_grad(self->get()));
}

// [[Rcpp::export]]
bool cpp_tensor_requires_grad (Rcpp::XPtr<XPtrTorchTensor> self) {
  return lantern_Tensor_requires_grad(self->get());
}

#include <thread>

void rcpp_call_hook (void* x, void* hook) {
  (*reinterpret_cast<std::function<void(void*)> *>(hook))(x);
}

// [[Rcpp::export]]
void cpp_tensor_register_hook (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Function f) {
  Rcpp::Rcout << std::this_thread::get_id() << std::endl;
  auto r_hook = (void *)new std::function<void(void *)>([f](void *x) {
    Rcpp::Rcout << std::this_thread::get_id() << std::endl;
    Rcpp::Rcout << "hey" << std::endl;
    auto y = make_xptr<XPtrTorchTensor>(x);
    Rcpp::Rcout << "hey2" << std::endl;
  });
  auto hook = lantern_new_hook(&rcpp_call_hook, r_hook);
  lantern_Tensor_register_hook(self->get(), hook);
}


