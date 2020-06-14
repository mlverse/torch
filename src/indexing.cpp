#include "torch_types.h"
#include "utils.hpp" 

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_torch_tensor_index(Rcpp::XPtr<XPtrTorchTensor> self, 
                                                   Rcpp::XPtr<XPtrTorchTensorIndex> index)
{
  XPtrTorchTensor out = lantern_Tensor_index(self->get(), index->get());
  return make_xptr<XPtrTorchTensor>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorIndex> cpp_torch_tensor_index_new()
{
  XPtrTorchTensorIndex out = lantern_TensorIndex_new();
  return make_xptr<XPtrTorchTensorIndex>(out);
}

// [[Rcpp::export]]
void cpp_torch_tensor_index_append_tensor(Rcpp::XPtr<XPtrTorchTensorIndex> self, Rcpp::XPtr<XPtrTorchTensor> x) {
  lantern_TensorIndex_append_tensor(self->get(), x->get());
}

// [[Rcpp::export]]
void cpp_torch_tensor_index_append_bool(Rcpp::XPtr<XPtrTorchTensorIndex> self, bool x) {
  lantern_TensorIndex_append_bool(self->get(), x);
}

// [[Rcpp::export]]
void cpp_torch_tensor_index_append_ellipsis(Rcpp::XPtr<XPtrTorchTensorIndex> self) {
  lantern_TensorIndex_append_ellipsis(self->get());
}

// [[Rcpp::export]]
void cpp_torch_tensor_index_append_none(Rcpp::XPtr<XPtrTorchTensorIndex> self) {
  lantern_TensorIndex_append_none(self->get());
}

// [[Rcpp::export]]
void cpp_torch_tensor_index_append_slice(Rcpp::XPtr<XPtrTorchTensorIndex> self, Rcpp::XPtr<XPtrTorch> x) {
  lantern_TensorIndex_append_slice(self->get(), x->get());
}