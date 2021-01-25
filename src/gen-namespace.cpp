// This file is auto generated. Dont modify it by hand.
#include "utils.h"

// [[Rcpp::export]]
void cpp_torch_method_set_data_self_Tensor_new_data_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> new_data) {
  lantern_Tensor_set_data_tensor_tensor(self->get(), new_data->get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_data_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_data_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_leaf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_leaf_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_output_nr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_output_nr_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method__version_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__version_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_requires_grad__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool requires_grad) {
  auto r_out = lantern_Tensor_requires_grad__tensor_bool(self->get(), reinterpret_cast<void*>(&requires_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_method_retain_grad_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  lantern_Tensor_retain_grad_tensor(self->get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rename__self_Tensor_names_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_rename__tensor_dimnamelist(self->get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rename_self_Tensor_names_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_rename_tensor_dimnamelist(self->get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_to_self_Tensor_names_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_align_to_tensor_dimnamelist(self->get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_to_self_Tensor_order_DimnameList_ellipsis_idx_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> order, nullable<int64_t> ellipsis_idx) {
  auto r_out = lantern_Tensor_align_to_tensor_dimnamelist_intt(self->get(), order->get(), ellipsis_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_as_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_align_as_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_refine_names_self_Tensor_names_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_refine_names_tensor_dimnamelist(self->get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_abs_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_abs_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_abs__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_abs__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_absolute_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_absolute_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_absolute__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_absolute__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_angle_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_angle_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sgn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sgn_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sgn__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sgn__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_conj_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_conj_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_acos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_acos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arccos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arccos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_add_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_add__tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_add_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_add__tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmv_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addmv_tensor_tensor_tensor_scalar_scalar(self->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmv__self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addmv__tensor_tensor_tensor_scalar_scalar(self->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addr_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec1, Rcpp::XPtr<XPtrTorchTensor> vec2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addr_tensor_tensor_tensor_scalar_scalar(self->get(), vec1->get(), vec2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addr__self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec1, Rcpp::XPtr<XPtrTorchTensor> vec2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addr__tensor_tensor_tensor_scalar_scalar(self->get(), vec1->get(), vec2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_all_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_all_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_allclose_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = lantern_Tensor_allclose_tensor_tensor_double_double_bool(self->get(), other->get(), XPtrTorch(lantern_double(rtol)).get(), XPtrTorch(lantern_double(atol)).get(), reinterpret_cast<void*>(&equal_nan));
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_any_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_any_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argmax_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_argmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argmin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_argmin_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_acosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_acosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arccosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arccosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_asinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_asinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arcsinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arcsinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_atanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_atanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arctanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arctanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, nullable<int64_t> storage_offset) {
  auto r_out = lantern_Tensor_as_strided_tensor_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, nullable<int64_t> storage_offset) {
  auto r_out = lantern_Tensor_as_strided__tensor_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_asin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_asin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arcsin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arcsin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_atan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_atan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arctan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_arctan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_baddbmm_tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_baddbmm__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_baddbmm__tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_bernoulli_tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli__self_Tensor_p_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_bernoulli__tensor_tensor_generator(self->get(), p->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_bernoulli__tensor_double_generator(self->get(), XPtrTorch(lantern_double(p)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli_self_Tensor_p_double (Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_bernoulli_tensor_double_generator(self->get(), XPtrTorch(lantern_double(p)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bincount_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weights, nullable<int64_t> minlength) {
  auto r_out = lantern_Tensor_bincount_tensor_tensor_intt(self->get(), weights->get(), minlength.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_not_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_bitwise_not_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_not__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_bitwise_not__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_not_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_logical_not_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_not__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_logical_not__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_xor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_xor_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_xor__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_xor__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_and_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_and_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_and__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_and__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_or_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_or_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_or__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logical_or__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bmm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_Tensor_bmm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ceil_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_ceil_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ceil__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_ceil__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_unsafe_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> chunks, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unsafe_chunk_tensor_intt_intt(self->get(), chunks.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> chunks, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_chunk_tensor_intt_intt(self->get(), chunks.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clamp_tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clamp__tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max_self_Tensor_max_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clamp_max_tensor_scalar(self->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max__self_Tensor_max_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clamp_max__tensor_scalar(self->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min_self_Tensor_min_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min) {
  auto r_out = lantern_Tensor_clamp_min_tensor_scalar(self->get(), min->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min__self_Tensor_min_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min) {
  auto r_out = lantern_Tensor_clamp_min__tensor_scalar(self->get(), min->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clip_tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_clip__tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_contiguous_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_contiguous_tensor_memoryformat(self->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copy__self_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> src, bool non_blocking) {
  auto r_out = lantern_Tensor_copy__tensor_tensor_bool(self->get(), src->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_cos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_cos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_cosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_cosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_count_nonzero_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim) {
  auto r_out = lantern_Tensor_count_nonzero_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_count_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_count_nonzero_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_cummax_tensor_intt(self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_cummax_tensor_dimname(self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummin_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_cummin_tensor_intt(self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummin_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_cummin_tensor_dimname(self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_cumprod_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_cumprod_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_cumsum_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_cumsum_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diag_embed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset, nullable<int64_t> dim1, nullable<int64_t> dim2) {
  auto r_out = lantern_Tensor_diag_embed_tensor_intt_intt_intt(self->get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagflat_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset) {
  auto r_out = lantern_Tensor_diagflat_tensor_intt(self->get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagonal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset, nullable<int64_t> dim1, nullable<int64_t> dim2) {
  auto r_out = lantern_Tensor_diagonal_tensor_intt_intt_intt(self->get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagonal_self_Tensor_outdim_Dimname_dim1_Dimname_dim2_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> outdim, Rcpp::XPtr<XPtrTorchDimname> dim1, Rcpp::XPtr<XPtrTorchDimname> dim2, nullable<int64_t> offset) {
  auto r_out = lantern_Tensor_diagonal_tensor_dimname_dimname_dimname_intt(self->get(), outdim->get(), dim1->get(), dim2->get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill_diagonal__self_Tensor_fill_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> fill_value, bool wrap) {
  auto r_out = lantern_Tensor_fill_diagonal__tensor_scalar_bool(self->get(), fill_value->get(), reinterpret_cast<void*>(&wrap));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_div_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_div__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_div_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_div__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_divide__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_divide__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_true_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_true_divide__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_true_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_true_divide__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dot_self_Tensor_tensor_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor) {
  auto r_out = lantern_Tensor_dot_tensor_tensor(self->get(), tensor->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_vdot_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_vdot_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_empty_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_Tensor_new_empty_tensor_intarrayref_tensoroptions(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_full_self_Tensor_size_IntArrayRef_fill_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchScalar> fill_value, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_Tensor_new_full_tensor_intarrayref_scalar_tensoroptions(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), fill_value->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_zeros_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_Tensor_new_zeros_tensor_intarrayref_tensoroptions(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resize__self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_resize__tensor_intarrayref_memoryformat(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erf__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erf__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erfc_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfc__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erfc__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_exp_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_exp__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp2_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_exp2_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp2__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_exp2__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expm1_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_expm1_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expm1__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_expm1__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expand_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, bool implicit) {
  auto r_out = lantern_Tensor_expand_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), reinterpret_cast<void*>(&implicit));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expand_as_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_expand_as_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> start_dim, nullable<int64_t> end_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_intt_intt(self->get(), start_dim.get(), end_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> start_dim, nullable<int64_t> end_dim, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_intt_intt_dimname(self->get(), start_dim.get(), end_dim.get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> start_dim, Rcpp::XPtr<XPtrTorchDimname> end_dim, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_dimname_dimname_dimname(self->get(), start_dim->get(), end_dim->get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dims, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_dimnamelist_dimname(self->get(), dims->get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unflatten_self_Tensor_dim_int64_t_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, std::vector<int64_t> sizes, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_unflatten_tensor_intt_intarrayref_dimnamelist(self->get(), dim.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(sizes.data(), sizes.size())).get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unflatten_self_Tensor_dim_Dimname_sizes_IntArrayRef_names_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, std::vector<int64_t> sizes, Rcpp::XPtr<XPtrTorch> names) {
  auto r_out = lantern_Tensor_unflatten_tensor_dimname_intarrayref_dimnamelist(self->get(), dim->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(sizes.data(), sizes.size())).get(), names->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill__self_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_fill__tensor_scalar(self->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill__self_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_fill__tensor_tensor(self->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_floor_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_floor__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_floor_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_floor_divide__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_floor_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_floor_divide__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_frac_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_frac_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_frac__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_frac__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gcd_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_gcd_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gcd__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_gcd__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lcm_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_lcm_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lcm__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_lcm__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ifft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized) {
  auto r_out = lantern_Tensor_ifft_tensor_intt_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized, bool onesided) {
  auto r_out = lantern_Tensor_rfft_tensor_intt_bool_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_irfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized, bool onesided, std::vector<int64_t> signal_sizes) {
  auto r_out = lantern_Tensor_irfft_tensor_intt_bool_bool_intarrayref(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), XPtrTorchvector_int64_t(lantern_vector_int64_t(signal_sizes.data(), signal_sizes.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_self_Tensor_indices_TensorList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices) {
  auto r_out = lantern_Tensor_index_tensor_tensorlist(self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_copy__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_copy_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy__self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_copy__tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_copy_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_put__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices, Rcpp::XPtr<XPtrTorchTensor> values, bool accumulate) {
  auto r_out = lantern_Tensor_index_put__tensor_tensorlist_tensor_bool(self->get(), indices->get(), values->get(), reinterpret_cast<void*>(&accumulate));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_put_self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices, Rcpp::XPtr<XPtrTorchTensor> values, bool accumulate) {
  auto r_out = lantern_Tensor_index_put_tensor_tensorlist_tensor_bool(self->get(), indices->get(), values->get(), reinterpret_cast<void*>(&accumulate));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_inverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_inverse_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isclose_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = lantern_Tensor_isclose_tensor_tensor_double_double_bool(self->get(), other->get(), XPtrTorch(lantern_double(rtol)).get(), XPtrTorch(lantern_double(atol)).get(), reinterpret_cast<void*>(&equal_nan));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isnan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isnan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_distributed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_distributed_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_floating_point_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_floating_point_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_complex_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_complex_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isreal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isreal_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_nonzero_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_same_size_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_is_same_size_tensor_tensor(self->get(), other->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_signed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_signed_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_kthvalue_tensor_intt_intt_bool(self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_kthvalue_tensor_intt_dimname_bool(self->get(), k.get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log10_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log10_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log10__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log10__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log1p_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log1p_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log1p__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log1p__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log2_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log2_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log2__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_log2__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logaddexp_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logaddexp_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logaddexp2_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_logaddexp2_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logdet_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_logdet_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_log_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_log_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logcumsumexp_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_logcumsumexp_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logcumsumexp_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_logcumsumexp_tensor_dimname(self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logsumexp_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_logsumexp_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logsumexp_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_Tensor_logsumexp_tensor_dimnamelist_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matmul_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_matmul_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matrix_power_self_Tensor_n_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n) {
  auto r_out = lantern_Tensor_matrix_power_tensor_intt(self->get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matrix_exp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_matrix_exp_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_max_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_max_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_amax_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_amax_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_mean_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_mean_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_mean_tensor_dimnamelist_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_median_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_median_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_min_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_min_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_amin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_amin_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_Tensor_mm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_mode_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_Tensor_mode_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_mul_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_mul__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_mul_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_mul__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_multiply_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_multiply__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_multiply_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_multiply__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mv_self_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec) {
  auto r_out = lantern_Tensor_mv_tensor_tensor(self->get(), vec->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mvlgamma_self_Tensor_p_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> p) {
  auto r_out = lantern_Tensor_mvlgamma_tensor_intt(self->get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mvlgamma__self_Tensor_p_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> p) {
  auto r_out = lantern_Tensor_mvlgamma__tensor_intt(self->get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_copy_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> length) {
  auto r_out = lantern_Tensor_narrow_copy_tensor_intt_intt_intt(self->get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> length) {
  auto r_out = lantern_Tensor_narrow_tensor_intt_intt_intt(self->get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_self_Tensor_dim_int64_t_start_Tensor_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> start, nullable<int64_t> length) {
  auto r_out = lantern_Tensor_narrow_tensor_intt_tensor_intt(self->get(), dim.get(), start->get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_permute_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dims) {
  auto r_out = lantern_Tensor_permute_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_movedim_self_Tensor_source_IntArrayRef_destination_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> source, std::vector<int64_t> destination) {
  auto r_out = lantern_Tensor_movedim_tensor_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(source.data(), source.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(destination.data(), destination.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_movedim_self_Tensor_source_int64_t_destination_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> source, nullable<int64_t> destination) {
  auto r_out = lantern_Tensor_movedim_tensor_intt_intt(self->get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_numpy_T_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_numpy_t_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_pinned_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_pinned_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pin_memory_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_pin_memory_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pinverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double rcond) {
  auto r_out = lantern_Tensor_pinverse_tensor_double(self->get(), XPtrTorch(lantern_double(rcond)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rad2deg_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_rad2deg_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rad2deg__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_rad2deg__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_deg2rad_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_deg2rad_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_deg2rad__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_deg2rad__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reciprocal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_reciprocal_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reciprocal__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_reciprocal__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_neg_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_neg_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_neg__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_neg__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_negative_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_negative_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_negative__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_negative__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_self_Tensor_repeats_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> repeats) {
  auto r_out = lantern_Tensor_repeat_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(repeats.data(), repeats.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_interleave_self_Tensor_repeats_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> repeats, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_repeat_interleave_tensor_tensor_intt(self->get(), repeats->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_interleave_self_Tensor_repeats_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> repeats, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_repeat_interleave_tensor_intt_intt(self->get(), repeats.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> shape) {
  auto r_out = lantern_Tensor_reshape_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(shape.data(), shape.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reshape_as_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_reshape_as_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_round_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_round_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_round__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_round__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_relu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_relu_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_relu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_relu__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prelu_self_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_Tensor_prelu_tensor_tensor(self->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_Tensor_prelu_backward_tensor_tensor_tensor(grad_output->get(), self->get(), weight->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hardshrink_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_Tensor_hardshrink_tensor_scalar(self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_Tensor_hardshrink_backward_tensor_tensor_scalar(grad_out->get(), self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rsqrt_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_rsqrt_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rsqrt__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_rsqrt__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_select_self_Tensor_dim_Dimname_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, nullable<int64_t> index) {
  auto r_out = lantern_Tensor_select_tensor_dimname_intt(self->get(), dim->get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_select_self_Tensor_dim_int64_t_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> index) {
  auto r_out = lantern_Tensor_select_tensor_intt_intt(self->get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sigmoid_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sigmoid_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sigmoid__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sigmoid__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logit_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_Tensor_logit_tensor_double(self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logit__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_Tensor_logit__tensor_double(self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_detach_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_detach_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_detach__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_detach__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_size_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_size_tensor_intt(self->get(), dim.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_size_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_size_tensor_dimname(self->get(), dim->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_slice_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> end, nullable<int64_t> step) {
  auto r_out = lantern_Tensor_slice_tensor_intt_intt_intt_intt(self->get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_slogdet_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_slogdet_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_smm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_Tensor_smm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_unsafe_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> split_size, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unsafe_split_tensor_intt_intt(self->get(), split_size.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> split_size, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_split_tensor_intt_intt(self->get(), split_size.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_unsafe_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> split_sizes, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unsafe_split_with_sizes_tensor_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(split_sizes.data(), split_sizes.size())).get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> split_sizes, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_split_with_sizes_tensor_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(split_sizes.data(), split_sizes.size())).get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_squeeze_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_squeeze_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_squeeze_tensor_dimname(self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_squeeze__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_squeeze__tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_squeeze__tensor_dimname(self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_sspaddmm_tensor_tensor_tensor_scalar_scalar(self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_stft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n_fft, nullable<int64_t> hop_length, nullable<int64_t> win_length, Rcpp::XPtr<XPtrTorchTensor> window, bool normalized, bool onesided, bool return_complex) {
  auto r_out = lantern_Tensor_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(self->get(), n_fft.get(), hop_length.get(), win_length.get(), window->get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), reinterpret_cast<void*>(&return_complex));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_istft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n_fft, nullable<int64_t> hop_length, nullable<int64_t> win_length, Rcpp::XPtr<XPtrTorchTensor> window, bool center, bool normalized, bool onesided, nullable<int64_t> length, bool return_complex) {
  auto r_out = lantern_Tensor_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(self->get(), n_fft.get(), hop_length.get(), win_length.get(), window->get(), reinterpret_cast<void*>(&center), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), length.get(), reinterpret_cast<void*>(&return_complex));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_stride_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_stride_tensor_intt(self->get(), dim.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_stride_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_stride_tensor_dimname(self->get(), dim->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_sum_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_sum_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_sum_tensor_dimnamelist_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nansum_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_nansum_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nansum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_nansum_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_to_size_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size) {
  auto r_out = lantern_Tensor_sum_to_size_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sqrt_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sqrt_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sqrt__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sqrt__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_square_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_square_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_square__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_square__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_Tensor_std_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_prod_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_prod_tensor_intt_bool_scalartype(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_prod_tensor_dimname_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_t_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_t__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_t__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_tan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_tan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_tanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_tanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim0, nullable<int64_t> dim1) {
  auto r_out = lantern_Tensor_transpose_tensor_intt_intt(self->get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim0, Rcpp::XPtr<XPtrTorchDimname> dim1) {
  auto r_out = lantern_Tensor_transpose_tensor_dimname_dimname(self->get(), dim0->get(), dim1->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim0, nullable<int64_t> dim1) {
  auto r_out = lantern_Tensor_transpose__tensor_intt_intt(self->get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flip_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dims) {
  auto r_out = lantern_Tensor_flip_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fliplr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_fliplr_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flipud_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_flipud_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_roll_self_Tensor_shifts_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
  auto r_out = lantern_Tensor_roll_tensor_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(shifts.data(), shifts.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rot90_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, std::vector<int64_t> dims) {
  auto r_out = lantern_Tensor_rot90_tensor_intt_intarrayref(self->get(), k.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trunc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_trunc_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trunc__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_trunc__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fix_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_fix_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fix__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_fix__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_type_as_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_type_as_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unsqueeze_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unsqueeze_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unsqueeze__self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unsqueeze__tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_Tensor_var_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_view_as_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_view_as_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_where_tensor_tensor_tensor(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_scalartype(self->get(), p->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p) {
  auto r_out = lantern_Tensor_norm_tensor_scalar(self->get(), p->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_intarrayref_bool_scalartype(self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_intarrayref_bool(self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_dimnamelist_bool_scalartype(self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_dimnamelist_bool(self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clone_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_clone_tensor_memoryformat(self->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resize_as__self_Tensor_the_template_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> the_template, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_resize_as__tensor_tensor_memoryformat(self->get(), the_template->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_zero__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_zero__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_sub_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_sub__tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_sub_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_sub__tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_subtract_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_subtract__tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_subtract_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_subtract__tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_heaviside_self_Tensor_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values) {
  auto r_out = lantern_Tensor_heaviside_tensor_tensor(self->get(), values->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_heaviside__self_Tensor_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values) {
  auto r_out = lantern_Tensor_heaviside__tensor_tensor(self->get(), values->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addmm_tensor_tensor_tensor_scalar_scalar(self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmm__self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addmm__tensor_tensor_tensor_scalar_scalar(self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_resize__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, nullable<int64_t> sparse_dim, nullable<int64_t> dense_dim) {
  auto r_out = lantern_Tensor_sparse_resize__tensor_intarrayref_intt_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), sparse_dim.get(), dense_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_resize_and_clear__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, nullable<int64_t> sparse_dim, nullable<int64_t> dense_dim) {
  auto r_out = lantern_Tensor_sparse_resize_and_clear__tensor_intarrayref_intt_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), sparse_dim.get(), dense_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_mask_self_Tensor_mask_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask) {
  auto r_out = lantern_Tensor_sparse_mask_tensor_tensor(self->get(), mask->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_dense_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_to_dense_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_sparse_dim_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sparse_dim_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method__dimI_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__dimi_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_dense_dim_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_dense_dim_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method__dimV_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__dimv_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method__nnz_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__nnz_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_coalesce_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_coalesce_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_coalesced_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_is_coalesced_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__indices_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__indices_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__values_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor__values_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__coalesced__self_Tensor_coalesced_bool (Rcpp::XPtr<XPtrTorchTensor> self, bool coalesced) {
  auto r_out = lantern_Tensor__coalesced__tensor_bool(self->get(), reinterpret_cast<void*>(&coalesced));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_indices_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_indices_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_values_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_values_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_unbind_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_unbind_tensor_intt(self->get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_unbind_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_Tensor_unbind_tensor_dimname(self->get(), dim->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_sparse_self_Tensor_sparse_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> sparse_dim) {
  auto r_out = lantern_Tensor_to_sparse_tensor_intt(self->get(), sparse_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_sparse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_to_sparse_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_mkldnn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_to_mkldnn_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dequantize_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_dequantize_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
double cpp_torch_method_q_scale_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_q_scale_tensor(self->get());
return reinterpret_and_clean<double, lantern_double_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_q_zero_point_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_q_zero_point_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_q_per_channel_scales_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_q_per_channel_scales_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_q_per_channel_zero_points_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_q_per_channel_zero_points_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_q_per_channel_axis_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_q_per_channel_axis_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_int_repr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_int_repr_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_method_qscheme_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_qscheme_tensor(self->get());
return make_xptr<XPtrTorchQScheme>(r_out, "QScheme");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, bool non_blocking, bool copy, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_to_tensor_tensoroptions_bool_bool_memoryformat(self->get(), options->get(), reinterpret_cast<void*>(&non_blocking), reinterpret_cast<void*>(&copy), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_device_Device_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDevice> device, Rcpp::XPtr<XPtrTorch> dtype, bool non_blocking, bool copy, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_to_tensor_device_scalartype_bool_bool_memoryformat(self->get(), device->get(), dtype->get(), reinterpret_cast<void*>(&non_blocking), reinterpret_cast<void*>(&copy), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype, bool non_blocking, bool copy, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_to_tensor_scalartype_bool_bool_memoryformat(self->get(), dtype->get(), reinterpret_cast<void*>(&non_blocking), reinterpret_cast<void*>(&copy), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, bool non_blocking, bool copy, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_Tensor_to_tensor_tensor_bool_bool_memoryformat(self->get(), other->get(), reinterpret_cast<void*>(&non_blocking), reinterpret_cast<void*>(&copy), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalar> cpp_torch_method_item_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_item_tensor(self->get());
return make_xptr<XPtrTorchScalar>(r_out, "Scalar");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Storage (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> source) {
  auto r_out = lantern_Tensor_set__tensor_storage(self->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Storage_storage_offset_int64_t_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> source, nullable<int64_t> storage_offset, std::vector<int64_t> size, std::vector<int64_t> stride) {
  auto r_out = lantern_Tensor_set__tensor_storage_intt_intarrayref_intarrayref(self->get(), source->get(), storage_offset.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_set__tensor_tensor(self->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_set__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_set_to_self_Tensor_tensor_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor) {
  auto r_out = lantern_Tensor_is_set_to_tensor_tensor(self->get(), tensor->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_masked_fill__tensor_tensor_scalar(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_masked_fill_tensor_tensor_scalar(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_masked_fill__tensor_tensor_tensor(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_masked_fill_tensor_tensor_tensor(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_scatter__self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_masked_scatter__tensor_tensor_tensor(self->get(), mask->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_masked_scatter_tensor_tensor_tensor(self->get(), mask->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_view_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size) {
  auto r_out = lantern_Tensor_view_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_put__self_Tensor_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source, bool accumulate) {
  auto r_out = lantern_Tensor_put__tensor_tensor_tensor_bool(self->get(), index->get(), source->get(), reinterpret_cast<void*>(&accumulate));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_add__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_add_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_Tensor_index_add_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_index_fill__tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_index_fill_tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_index_fill__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_index_fill_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_index_fill__tensor_dimname_tensor_scalar(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_index_fill__tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_index_fill_tensor_dimname_tensor_scalar(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_Tensor_index_fill_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_scatter_tensor_dimname_tensor_scalar(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_src_Tensor_reduce_stdstring (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src, std::string reduce) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_tensor_stdstring(self->get(), dim.get(), index->get(), src->get(), XPtrTorchstring(lantern_string_new(reduce.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_value_Scalar_reduce_stdstring (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value, std::string reduce) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_scalar_stdstring(self->get(), dim.get(), index->get(), value->get(), XPtrTorchstring(lantern_string_new(reduce.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter_add__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter_add_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_Tensor_scatter_add_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_eq__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_eq__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_and_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_and_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_and__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_and__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___and___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___and___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___and___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___and___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___iand___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___iand___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___iand___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___iand___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_or_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_or_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_or__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_or__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___or___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___or___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___or___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___or___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ior___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___ior___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ior___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___ior___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_xor_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_xor_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_bitwise_xor__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_bitwise_xor__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___xor___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___xor___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___xor___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___xor___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ixor___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___ixor___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ixor___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___ixor___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___lshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___lshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___lshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___lshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ilshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___ilshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ilshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___ilshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___rshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___rshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___rshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___rshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___irshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor___irshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___irshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor___irshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lgamma__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_lgamma__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan2__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_atan2__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tril__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_Tensor_tril__tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_triu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_Tensor_triu__tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_digamma__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_digamma__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_polygamma__self_Tensor_n_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n) {
  auto r_out = lantern_Tensor_polygamma__tensor_intt(self->get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_renorm__self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchScalar> maxnorm) {
  auto r_out = lantern_Tensor_renorm__tensor_scalar_intt_scalar(self->get(), p->get(), dim.get(), maxnorm->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow__self_Tensor_exponent_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> exponent) {
  auto r_out = lantern_Tensor_pow__tensor_scalar(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow__self_Tensor_exponent_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_Tensor_pow__tensor_tensor(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchScalar> weight) {
  auto r_out = lantern_Tensor_lerp__tensor_tensor_scalar(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_Tensor_lerp__tensor_tensor_tensor(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_fmod__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_fmod__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_remainder__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_remainder__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addbmm__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addbmm__tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_Tensor_addbmm_tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcdiv__self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_addcdiv__tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor_from_int64_t_to_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> from, nullable<int64_t> to, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_random__tensor_intt_intt_generator(self->get(), from.get(), to.get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor_to_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> to, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_random__tensor_intt_generator(self->get(), to.get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_random__tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_uniform__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double from, double to, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_uniform__tensor_double_double_generator(self->get(), XPtrTorch(lantern_double(from)).get(), XPtrTorch(lantern_double(to)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cauchy__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double median, double sigma, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_cauchy__tensor_double_double_generator(self->get(), XPtrTorch(lantern_double(median)).get(), XPtrTorch(lantern_double(sigma)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_normal__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double mean, double std, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_log_normal__tensor_double_double_generator(self->get(), XPtrTorch(lantern_double(mean)).get(), XPtrTorch(lantern_double(std)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exponential__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double lambd, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_exponential__tensor_double_generator(self->get(), XPtrTorch(lantern_double(lambd)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_geometric__self_Tensor_p_double (Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_geometric__tensor_double_generator(self->get(), XPtrTorch(lantern_double(p)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diag_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_Tensor_diag_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cross_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, nullable<int64_t> dim) {
  auto r_out = lantern_Tensor_cross_tensor_tensor_intt(self->get(), other->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_triu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_Tensor_triu_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tril_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_Tensor_tril_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trace_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_trace_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_ne_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_ne_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_ne__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_ne__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_not_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_not_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_not_equal__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_not_equal__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_eq_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_eq_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_ge_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_ge_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_ge__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_ge__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_greater_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_greater_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_greater_equal__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_greater_equal__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_le_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_le_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_le__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_le__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_less_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_less_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_less_equal__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_less_equal__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_gt_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_gt_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_gt__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_gt__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_greater_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_greater_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_greater__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_greater__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_lt_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_lt_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_lt__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_lt__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_less_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_less_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less__self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_less__tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_less__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_take_self_Tensor_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_Tensor_take_tensor_tensor(self->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_select_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_Tensor_index_select_tensor_intt_tensor(self->get(), dim.get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_select_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_Tensor_index_select_tensor_dimname_tensor(self->get(), dim->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_select_self_Tensor_mask_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask) {
  auto r_out = lantern_Tensor_masked_select_tensor_tensor(self->get(), mask->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_nonzero_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_method_nonzero_numpy_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_nonzero_numpy_tensor(self->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gather_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_Tensor_gather_tensor_intt_tensor_bool(self->get(), dim.get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gather_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_Tensor_gather_tensor_dimname_tensor_bool(self->get(), dim->get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_addcmul_tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcmul__self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_addcmul__tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_Tensor_addcdiv_tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_lstsq_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_Tensor_lstsq_tensor_tensor(self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_triangular_solve_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = lantern_Tensor_triangular_solve_tensor_tensor_bool_bool_bool(self->get(), A->get(), reinterpret_cast<void*>(&upper), reinterpret_cast<void*>(&transpose), reinterpret_cast<void*>(&unitriangular));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_symeig_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors, bool upper) {
  auto r_out = lantern_Tensor_symeig_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&eigenvectors), reinterpret_cast<void*>(&upper));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_eig_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors) {
  auto r_out = lantern_Tensor_eig_tensor_bool(self->get(), reinterpret_cast<void*>(&eigenvectors));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_svd_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool some, bool compute_uv) {
  auto r_out = lantern_Tensor_svd_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&some), reinterpret_cast<void*>(&compute_uv));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_Tensor_cholesky_tensor_bool(self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_solve_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, bool upper) {
  auto r_out = lantern_Tensor_cholesky_solve_tensor_tensor_bool(self->get(), input2->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_solve_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_Tensor_solve_tensor_tensor(self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_inverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_Tensor_cholesky_inverse_tensor_bool(self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_qr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool some) {
  auto r_out = lantern_Tensor_qr_tensor_bool(self->get(), reinterpret_cast<void*>(&some));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_geqrf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_geqrf_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_orgqr_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2) {
  auto r_out = lantern_Tensor_orgqr_tensor_tensor(self->get(), input2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ormqr_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> input3, bool left, bool transpose) {
  auto r_out = lantern_Tensor_ormqr_tensor_tensor_tensor_bool_bool(self->get(), input2->get(), input3->get(), reinterpret_cast<void*>(&left), reinterpret_cast<void*>(&transpose));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> LU_data, Rcpp::XPtr<XPtrTorchTensor> LU_pivots) {
  auto r_out = lantern_Tensor_lu_solve_tensor_tensor_tensor(self->get(), LU_data->get(), LU_pivots->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multinomial_self_Tensor_num_samples_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> num_samples, bool replacement, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_multinomial_tensor_intt_bool_generator(self->get(), num_samples.get(), reinterpret_cast<void*>(&replacement), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lgamma_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_lgamma_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_digamma_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_digamma_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfinv_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erfinv_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfinv__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_erfinv__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_i0_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_i0_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_i0__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_i0__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sign_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sign_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sign__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_sign__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_signbit_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_signbit_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dist_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> p) {
  auto r_out = lantern_Tensor_dist_tensor_tensor_scalar(self->get(), other->get(), p->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan2_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_atan2_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchScalar> weight) {
  auto r_out = lantern_Tensor_lerp_tensor_tensor_scalar(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_Tensor_lerp_tensor_tensor_tensor(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_histc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> bins, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_Tensor_histc_tensor_intt_scalar_scalar(self->get(), bins.get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_fmod_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_fmod_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hypot_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_hypot_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hypot__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_hypot__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nextafter_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_nextafter_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nextafter__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_nextafter__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_Tensor_remainder_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_remainder_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_min_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_min_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_max_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_max_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_maximum_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_maximum_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_max_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_max_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_minimum_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_minimum_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_min_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_min_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_median_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_median_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_quantile_tensor_double_intt_bool(self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_quantile_tensor_tensor_intt_bool(self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_nanquantile_tensor_double_intt_bool(self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_Tensor_nanquantile_tensor_tensor_intt_bool(self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool descending) {
  auto r_out = lantern_Tensor_sort_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool descending) {
  auto r_out = lantern_Tensor_sort_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argsort_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool descending) {
  auto r_out = lantern_Tensor_argsort_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&descending));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argsort_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool descending) {
  auto r_out = lantern_Tensor_argsort_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&descending));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_topk_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool largest, bool sorted) {
  auto r_out = lantern_Tensor_topk_tensor_intt_intt_bool_bool(self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&largest), reinterpret_cast<void*>(&sorted));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_all_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_any_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchScalar> maxnorm) {
  auto r_out = lantern_Tensor_renorm_tensor_scalar_intt_scalar(self->get(), p->get(), dim.get(), maxnorm->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unfold_self_Tensor_dimension_int64_t_size_int64_t_step_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dimension, nullable<int64_t> size, nullable<int64_t> step) {
  auto r_out = lantern_Tensor_unfold_tensor_intt_intt_intt(self->get(), dimension.get(), size.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_Tensor_equal_tensor_tensor(self->get(), other->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow_self_Tensor_exponent_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_Tensor_pow_tensor_tensor(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow_self_Tensor_exponent_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> exponent) {
  auto r_out = lantern_Tensor_pow_tensor_scalar(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_normal__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double mean, double std, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_Tensor_normal__tensor_double_double_generator(self->get(), XPtrTorch(lantern_double(mean)).get(), XPtrTorch(lantern_double(std)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_alias_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_alias_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isfinite_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isfinite_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isinf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isinf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isposinf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isposinf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isneginf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_isneginf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized) {
  auto r_out = lantern_Tensor_fft_tensor_intt_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_det_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_Tensor_det_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_outer_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_Tensor_outer_tensor_tensor(self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ger_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_Tensor_ger_tensor_tensor(self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Byte_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_byte_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Char_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_char_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Double_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_double_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Float_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_float_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Int_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_int_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Long_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_long_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Short_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_short_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Half_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool non_blocking) {
  auto r_out = lantern__cast_half_tensor_bool(self->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_align_tensors_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_align_tensors_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
bool cpp_torch_namespace__use_cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t (Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, nullable<int64_t> blank) {
  auto r_out = lantern__use_cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt(log_probs->get(), targets->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_lengths.data(), input_lengths.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(target_lengths.data(), target_lengths.size())).get(), blank.get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t_deterministic_bool_zero_infinity_bool (Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, nullable<int64_t> blank, bool deterministic, bool zero_infinity) {
  auto r_out = lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool(log_probs->get(), targets->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_lengths.data(), input_lengths.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(target_lengths.data(), target_lengths.size())).get(), blank.get(), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&zero_infinity));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cudnn_rnn_flatten_weight_weight_arr_TensorList_weight_stride0_int64_t_input_size_int64_t_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensorList> weight_arr, nullable<int64_t> weight_stride0, nullable<int64_t> input_size, nullable<int64_t> mode, nullable<int64_t> hidden_size, nullable<int64_t> num_layers, bool batch_first, bool bidirectional) {
  auto r_out = lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_bool_bool(weight_arr->get(), weight_stride0.get(), input_size.get(), mode.get(), hidden_size.get(), num_layers.get(), reinterpret_cast<void*>(&batch_first), reinterpret_cast<void*>(&bidirectional));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> weight, nullable<int64_t> weight_stride0, Rcpp::XPtr<XPtrTorchTensor> weight_buf, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> cx, nullable<int64_t> mode, nullable<int64_t> hidden_size, nullable<int64_t> num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> dropout_state) {
  auto r_out = lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(input->get(), weight->get(), weight_stride0.get(), weight_buf->get(), hx->get(), cx->get(), mode.get(), hidden_size.get(), num_layers.get(), reinterpret_cast<void*>(&batch_first), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), XPtrTorchvector_int64_t(lantern_vector_int64_t(batch_sizes.data(), batch_sizes.size())).get(), dropout_state->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> weight, nullable<int64_t> weight_stride0, Rcpp::XPtr<XPtrTorchTensor> weight_buf, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> cx, Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> grad_cy, nullable<int64_t> mode, nullable<int64_t> hidden_size, nullable<int64_t> num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> dropout_state, Rcpp::XPtr<XPtrTorchTensor> reserve, std::vector<bool> output_mask) {
  auto r_out = lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(input->get(), weight->get(), weight_stride0.get(), weight_buf->get(), hx->get(), cx->get(), output->get(), grad_output->get(), grad_hy->get(), grad_cy->get(), mode.get(), hidden_size.get(), num_layers.get(), reinterpret_cast<void*>(&batch_first), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), XPtrTorchvector_int64_t(lantern_vector_int64_t(batch_sizes.data(), batch_sizes.size())).get(), dropout_state->get(), reserve->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),make_xptr<XPtrTorchTensorList>(lantern_vector_get(r_out, 3), "TensorList"));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cudnn_init_dropout_state_dropout_double_train_bool_dropout_seed_int64_t_options_TensorOptions (double dropout, bool train, nullable<int64_t> dropout_seed, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions(XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), dropout_seed.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__debug_has_internal_overlap_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__debug_has_internal_overlap_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fused_dropout_self_Tensor_p_double (Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern__fused_dropout_tensor_double_generator(self->get(), XPtrTorch(lantern_double(p)).get(), generator->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__masked_scale_self_Tensor_mask_Tensor_scale_double (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, double scale) {
  auto r_out = lantern__masked_scale_tensor_tensor_double(self->get(), mask->get(), XPtrTorch(lantern_double(scale)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__sobol_engine_draw_quasi_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> quasi, nullable<int64_t> n, Rcpp::XPtr<XPtrTorchTensor> sobolstate, nullable<int64_t> dimension, nullable<int64_t> num_generated, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype(quasi->get(), n.get(), sobolstate->get(), dimension.get(), num_generated.get(), dtype->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_ff__self_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, Rcpp::XPtr<XPtrTorchTensor> sobolstate, nullable<int64_t> dimension, nullable<int64_t> num_generated) {
  auto r_out = lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(self->get(), n.get(), sobolstate->get(), dimension.get(), num_generated.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_scramble__self_Tensor_ltm_Tensor_dimension_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> ltm, nullable<int64_t> dimension) {
  auto r_out = lantern__sobol_engine_scramble__tensor_tensor_intt(self->get(), ltm->get(), dimension.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_initialize_state__self_Tensor_dimension_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dimension) {
  auto r_out = lantern__sobol_engine_initialize_state__tensor_intt(self->get(), dimension.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__reshape_from_tensor_self_Tensor_shape_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> shape) {
  auto r_out = lantern__reshape_from_tensor_tensor_tensor(self->get(), shape->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__shape_as_tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__shape_as_tensor_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> input, double p, bool train) {
  auto r_out = lantern_dropout_tensor_double_bool(input->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> self, double p, bool train) {
  auto r_out = lantern_dropout__tensor_double_bool(self->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> input, double p, bool train) {
  auto r_out = lantern_feature_dropout_tensor_double_bool(input->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> self, double p, bool train) {
  auto r_out = lantern_feature_dropout__tensor_double_bool(self->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alpha_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> input, double p, bool train) {
  auto r_out = lantern_alpha_dropout_tensor_double_bool(input->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alpha_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> self, double p, bool train) {
  auto r_out = lantern_alpha_dropout__tensor_double_bool(self->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_alpha_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> input, double p, bool train) {
  auto r_out = lantern_feature_alpha_dropout_tensor_double_bool(input->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_alpha_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<XPtrTorchTensor> self, double p, bool train) {
  auto r_out = lantern_feature_alpha_dropout__tensor_double_bool(self->get(), XPtrTorch(lantern_double(p)).get(), reinterpret_cast<void*>(&train));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_abs_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_abs__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_abs_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_absolute_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_absolute_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_absolute_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_absolute_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_angle_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_angle_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_angle_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_angle_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_view_as_real_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_view_as_real_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_view_as_complex_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_view_as_complex_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sgn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sgn_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sgn_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sgn_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_real_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_real_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_imag_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_imag_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_conj_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_conj_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__conj_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__conj_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acos_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccos_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool1d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad) {
  auto r_out = lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_avg_pool1d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_max_pool1d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_add_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_add_out_tensor_tensor_tensor_scalar(out->get(), self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__add_relu_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__add_relu__tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__add_relu_out_tensor_tensor_tensor_scalar(out->get(), self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_add_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addmv_tensor_tensor_tensor_scalar_scalar(self->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv__self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addmv__tensor_tensor_tensor_scalar_scalar(self->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv_out_out_Tensor_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__addmv_impl__self_Tensor_self2_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> self2, Rcpp::XPtr<XPtrTorchTensor> mat, Rcpp::XPtr<XPtrTorchTensor> vec, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__addmv_impl__tensor_tensor_tensor_tensor_scalar_scalar(self->get(), self2->get(), mat->get(), vec->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addr_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec1, Rcpp::XPtr<XPtrTorchTensor> vec2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addr_tensor_tensor_tensor_scalar_scalar(self->get(), vec1->get(), vec2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addr_out_out_Tensor_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec1, Rcpp::XPtr<XPtrTorchTensor> vec2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), vec1->get(), vec2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_affine_grid_generator_theta_Tensor_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> theta, std::vector<int64_t> size, bool align_corners) {
  auto r_out = lantern_affine_grid_generator_tensor_intarrayref_bool(theta->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_affine_grid_generator_backward_grad_Tensor_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> size, bool align_corners) {
  auto r_out = lantern_affine_grid_generator_backward_tensor_intarrayref_bool(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_all_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_all_out_tensor_tensor_intt_bool(out->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_all_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_all_out_tensor_tensor_dimname_bool(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_allclose_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = lantern_allclose_tensor_tensor_double_double_bool(self->get(), other->get(), XPtrTorch(lantern_double(rtol)).get(), XPtrTorch(lantern_double(atol)).get(), reinterpret_cast<void*>(&equal_nan));
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_any_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_any_out_tensor_tensor_intt_bool(out->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_any_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_any_out_tensor_tensor_dimname_bool(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_end_Scalar (Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_arange_scalar_tensoroptions(end->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_arange_scalar_scalar_tensoroptions(start->get(), end->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_start_Scalar_end_Scalar_step_Scalar (Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchScalar> step, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_arange_scalar_scalar_scalar_tensoroptions(start->get(), end->get(), step->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_out_out_Tensor_end_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> end) {
  auto r_out = lantern_arange_out_tensor_scalar(out->get(), end->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchScalar> step) {
  auto r_out = lantern_arange_out_tensor_scalar_scalar_scalar(out->get(), start->get(), end->get(), step->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__dim_arange_like_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> like, nullable<int64_t> dim) {
  auto r_out = lantern__dim_arange_tensor_intt(like->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmax_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_argmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_argmin_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_acosh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arccosh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asinh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsinh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atanh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctanh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, nullable<int64_t> storage_offset) {
  auto r_out = lantern_as_strided_tensor_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, nullable<int64_t> storage_offset) {
  auto r_out = lantern_as_strided__tensor_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_asin_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arcsin_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atan_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_arctan_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_1d_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atleast_1d_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_atleast_1d_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_atleast_1d_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_2d_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atleast_2d_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_atleast_2d_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_atleast_2d_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_3d_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_atleast_3d_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_atleast_3d_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_atleast_3d_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_baddbmm_tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__baddbmm_mkl__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_baddbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bartlett_window_window_length_int64_t (nullable<int64_t> window_length, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_bartlett_window_intt_tensoroptions(window_length.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bartlett_window_window_length_int64_t_periodic_bool (nullable<int64_t> window_length, bool periodic, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_bartlett_window_intt_bool_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&cudnn_enabled));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_var_Tensor_eps_double_output_scale_double_output_zero_point_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> var, double eps, double output_scale, nullable<int64_t> output_zero_point) {
  auto r_out = lantern_quantized_batch_norm_tensor_tensor_tensor_tensor_tensor_double_double_intt(input->get(), weight->get(), bias->get(), mean->get(), var->get(), XPtrTorch(lantern_double(eps)).get(), XPtrTorch(lantern_double(output_scale)).get(), output_zero_point.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&cudnn_enabled));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),reinterpret_and_clean<int64_t, lantern_int64_t_delete>(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_backward_impl_index_int64_t_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_transform_Tensor_train_bool_eps_double_output_mask_stdarraybool3_reservedSpace_Tensor (nullable<int64_t> impl_index, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, Rcpp::XPtr<XPtrTorchTensor> save_mean, Rcpp::XPtr<XPtrTorchTensor> save_var_transform, bool train, double eps, std::vector<bool> output_mask, Rcpp::XPtr<XPtrTorchTensor> reservedSpace) {
  auto r_out = lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool_tensor(impl_index.get(), input->get(), grad_output->get(), weight->get(), running_mean->get(), running_var->get(), save_mean->get(), save_var_transform->get(), reinterpret_cast<void*>(&train), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&output_mask), reservedSpace->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_bernoulli_tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_bernoulli_out_tensor_tensor_generator(out->get(), self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_self_Tensor_p_double (Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_bernoulli_tensor_double_generator(self->get(), XPtrTorch(lantern_double(p)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bilinear_input1_Tensor_input2_Tensor_weight_Tensor_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> input1, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_bilinear_tensor_tensor_tensor_tensor(input1->get(), input2->get(), weight->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_tensor_tensor_tensor_intt(self->get(), target->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt(out->get(), self->get(), target->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt(grad_output->get(), self->get(), target->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt(grad_input->get(), grad_output->get(), self->get(), target->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_with_logits_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> pos_weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(self->get(), target->get(), weight->get(), pos_weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_with_logits_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> pos_weight, nullable<int64_t> reduction) {
  auto r_out = lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(grad_output->get(), self->get(), target->get(), weight->get(), pos_weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bincount_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weights, nullable<int64_t> minlength) {
  auto r_out = lantern_bincount_tensor_tensor_intt(self->get(), weights->get(), minlength.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_not_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_bitwise_not_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_not_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_bitwise_not_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_not_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_logical_not_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_not_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_logical_not_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_xor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_xor_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_xor_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_xor_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_and_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_and_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_and_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_and_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_or_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_or_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_or_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logical_or_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_blackman_window_window_length_int64_t (nullable<int64_t> window_length, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_blackman_window_intt_tensoroptions(window_length.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_blackman_window_window_length_int64_t_periodic_bool (nullable<int64_t> window_length, bool periodic, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_blackman_window_intt_bool_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bmm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_bmm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__bmm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2, bool deterministic) {
  auto r_out = lantern__bmm_tensor_tensor_bool(self->get(), mat2->get(), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bmm_out_out_Tensor_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_bmm_out_tensor_tensor_tensor(out->get(), self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__bmm_out_out_Tensor_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2, bool deterministic) {
  auto r_out = lantern__bmm_out_tensor_tensor_tensor_bool(out->get(), self->get(), mat2->get(), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_broadcast_tensors_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_broadcast_tensors_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern_cat_tensorlist_intt(tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern_cat_out_tensor_tensorlist_intt(out->get(), tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_tensors_TensorList_dim_Dimname (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cat_tensorlist_dimname(tensors->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cat_out_tensor_tensorlist_dimname(out->get(), tensors->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_block_diag_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_block_diag_tensorlist(tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_ceil_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_ceil__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_ceil_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_chain_matmul_matrices_TensorList (Rcpp::XPtr<XPtrTorchTensorList> matrices) {
  auto r_out = lantern_chain_matmul_tensorlist(matrices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_unsafe_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> chunks, nullable<int64_t> dim) {
  auto r_out = lantern_unsafe_chunk_tensor_intt_intt(self->get(), chunks.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> chunks, nullable<int64_t> dim) {
  auto r_out = lantern_chunk_tensor_intt_intt(self->get(), chunks.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp_tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp__tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp_out_tensor_tensor_scalar_scalar(out->get(), self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_self_Tensor_max_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp_max_tensor_scalar(self->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max__self_Tensor_max_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp_max__tensor_scalar(self->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_out_out_Tensor_self_Tensor_max_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clamp_max_out_tensor_tensor_scalar(out->get(), self->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_self_Tensor_min_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min) {
  auto r_out = lantern_clamp_min_tensor_scalar(self->get(), min->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min__self_Tensor_min_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min) {
  auto r_out = lantern_clamp_min__tensor_scalar(self->get(), min->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_out_out_Tensor_self_Tensor_min_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min) {
  auto r_out = lantern_clamp_min_out_tensor_tensor_scalar(out->get(), self->get(), min->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clip_tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clip__tensor_scalar_scalar(self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_clip_out_tensor_tensor_scalar_scalar(out->get(), self->get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_cudnn_is_acceptable_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cudnn_is_acceptable_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_complex_real_Tensor_imag_Tensor (Rcpp::XPtr<XPtrTorchTensor> real, Rcpp::XPtr<XPtrTorchTensor> imag) {
  auto r_out = lantern_complex_tensor_tensor(real->get(), imag->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_complex_out_out_Tensor_real_Tensor_imag_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> real, Rcpp::XPtr<XPtrTorchTensor> imag) {
  auto r_out = lantern_complex_out_tensor_tensor_tensor(out->get(), real->get(), imag->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polar_abs_Tensor_angle_Tensor (Rcpp::XPtr<XPtrTorchTensor> abs, Rcpp::XPtr<XPtrTorchTensor> angle) {
  auto r_out = lantern_polar_tensor_tensor(abs->get(), angle->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polar_out_out_Tensor_abs_Tensor_angle_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> abs, Rcpp::XPtr<XPtrTorchTensor> angle) {
  auto r_out = lantern_polar_out_tensor_tensor_tensor(out->get(), abs->get(), angle->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_constant_pad_nd_self_Tensor_pad_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> pad, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_constant_pad_nd_tensor_intarrayref_scalar(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(pad.data(), pad.size())).get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups) {
  auto r_out = lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_convolution_overrideable_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups) {
  auto r_out = lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_convolution_backward_overrideable_grad_output_Tensor_input_Tensor_weight_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups, std::vector<bool> output_mask) {
  auto r_out = lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool(grad_output->get(), input->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool_allow_tf32_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  auto r_out = lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&cudnn_enabled), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  auto r_out = lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&cudnn_enabled));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_nogroup_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding) {
  auto r_out = lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__convolution_double_backward_ggI_Tensor_ggW_Tensor_ggb_Tensor_gO_Tensor_weight_Tensor_self_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool_allow_tf32_bool_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> ggI, Rcpp::XPtr<XPtrTorchTensor> ggW, Rcpp::XPtr<XPtrTorchTensor> ggb, Rcpp::XPtr<XPtrTorchTensor> gO, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, nullable<int64_t> groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool_stdarraybool(ggI->get(), ggW->get(), ggb->get(), gO->get(), weight->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&transposed), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&cudnn_enabled), reinterpret_cast<void*>(&allow_tf32), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv1d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv2d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv3d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_tbc_self_Tensor_weight_Tensor_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, nullable<int64_t> pad) {
  auto r_out = lantern_conv_tbc_tensor_tensor_tensor_intt(self->get(), weight->get(), bias->get(), pad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_conv_tbc_backward_self_Tensor_input_Tensor_weight_Tensor_bias_Tensor_pad_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, nullable<int64_t> pad) {
  auto r_out = lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt(self->get(), input->get(), weight->get(), bias->get(), pad.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose1d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, nullable<int64_t> groups, std::vector<int64_t> dilation) {
  auto r_out = lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose2d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, nullable<int64_t> groups, std::vector<int64_t> dilation) {
  auto r_out = lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose3d_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, nullable<int64_t> groups, std::vector<int64_t> dilation) {
  auto r_out = lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), groups.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__copy_from_self_Tensor_dst_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> dst, bool non_blocking) {
  auto r_out = lantern__copy_from_tensor_tensor_bool(self->get(), dst->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cos_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cos__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cos_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cosh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cosh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_cosh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosine_embedding_loss_input1_Tensor_input2_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> input1, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> target, double margin, nullable<int64_t> reduction) {
  auto r_out = lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(input1->get(), input2->get(), target->get(), XPtrTorch(lantern_double(margin)).get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_count_nonzero_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim) {
  auto r_out = lantern_count_nonzero_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_count_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_count_nonzero_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_affine_grid_generator_theta_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (Rcpp::XPtr<XPtrTorchTensor> theta, nullable<int64_t> False, nullable<int64_t> C, nullable<int64_t> H, nullable<int64_t> W) {
  auto r_out = lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt(theta->get(), False.get(), C.get(), H.get(), W.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_affine_grid_generator_backward_grad_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, nullable<int64_t> False, nullable<int64_t> C, nullable<int64_t> H, nullable<int64_t> W) {
  auto r_out = lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt(grad->get(), False.get(), C.get(), H.get(), W.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double exponential_average_factor, double epsilon) {
  auto r_out = lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(exponential_average_factor)).get(), XPtrTorch(lantern_double(epsilon)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double_reserveSpace_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, Rcpp::XPtr<XPtrTorchTensor> save_mean, Rcpp::XPtr<XPtrTorchTensor> save_var, double epsilon, Rcpp::XPtr<XPtrTorchTensor> reserveSpace) {
  auto r_out = lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_tensor(input->get(), grad_output->get(), weight->get(), running_mean->get(), running_var->get(), save_mean->get(), save_var->get(), XPtrTorch(lantern_double(epsilon)).get(), reserveSpace->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (std::vector<int64_t> self_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(self_size.data(), self_size.size())).get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool_output_mask_stdarraybool2 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool_output_mask_stdarraybool2 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&allow_tf32));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_grid_sampler_self_Tensor_grid_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grid) {
  auto r_out = lantern_cudnn_grid_sampler_tensor_tensor(self->get(), grid->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_grid_sampler_backward_self_Tensor_grid_Tensor_grad_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grid, Rcpp::XPtr<XPtrTorchTensor> grad_output) {
  auto r_out = lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor(self->get(), grid->get(), grad_output->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_cummax_tensor_intt(self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_cummax_out_tensor_tensor_tensor_intt(values->get(), indices->get(), self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cummax_tensor_dimname(self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cummax_out_tensor_tensor_tensor_dimname(values->get(), indices->get(), self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
void cpp_torch_namespace__cummax_helper_self_Tensor_values_Tensor_indices_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> dim) {
  lantern__cummax_helper_tensor_tensor_tensor_intt(self->get(), values->get(), indices->get(), dim.get());
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_cummin_tensor_intt(self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_cummin_out_tensor_tensor_tensor_intt(values->get(), indices->get(), self->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cummin_tensor_dimname(self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_cummin_out_tensor_tensor_tensor_dimname(values->get(), indices->get(), self->get(), dim->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
void cpp_torch_namespace__cummin_helper_self_Tensor_values_Tensor_indices_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> dim) {
  lantern__cummin_helper_tensor_tensor_tensor_intt(self->get(), values->get(), indices->get(), dim.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cummaxmin_backward_grad_Tensor_input_Tensor_indices_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> dim) {
  auto r_out = lantern_cummaxmin_backward_tensor_tensor_tensor_intt(grad->get(), input->get(), indices->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumprod_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumprod_out_tensor_tensor_intt_scalartype(out->get(), self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumprod_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumprod_out_tensor_tensor_dimname_scalartype(out->get(), self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_backward_grad_Tensor_input_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input, nullable<int64_t> dim) {
  auto r_out = lantern_cumprod_backward_tensor_tensor_intt(grad->get(), input->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumsum_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumsum_out_tensor_tensor_intt_scalartype(out->get(), self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumsum_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_cumsum_out_tensor_tensor_dimname_scalartype(out->get(), self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, nullable<int64_t> blank, nullable<int64_t> reduction, bool zero_infinity) {
  auto r_out = lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(log_probs->get(), targets->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_lengths.data(), input_lengths.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(target_lengths.data(), target_lengths.size())).get(), blank.get(), reduction.get(), reinterpret_cast<void*>(&zero_infinity));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_Tensor_target_lengths_Tensor (Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, Rcpp::XPtr<XPtrTorchTensor> input_lengths, Rcpp::XPtr<XPtrTorchTensor> target_lengths, nullable<int64_t> blank, nullable<int64_t> reduction, bool zero_infinity) {
  auto r_out = lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(log_probs->get(), targets->get(), input_lengths->get(), target_lengths->get(), blank.get(), reduction.get(), reinterpret_cast<void*>(&zero_infinity));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, nullable<int64_t> blank, bool zero_infinity) {
  auto r_out = lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool(log_probs->get(), targets->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_lengths.data(), input_lengths.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(target_lengths.data(), target_lengths.size())).get(), blank.get(), reinterpret_cast<void*>(&zero_infinity));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__ctc_loss_backward_grad_Tensor_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_neg_log_likelihood_Tensor_log_alpha_Tensor_blank_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> log_probs, Rcpp::XPtr<XPtrTorchTensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, Rcpp::XPtr<XPtrTorchTensor> neg_log_likelihood, Rcpp::XPtr<XPtrTorchTensor> log_alpha, nullable<int64_t> blank, bool zero_infinity) {
  auto r_out = lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool(grad->get(), log_probs->get(), targets->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_lengths.data(), input_lengths.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(target_lengths.data(), target_lengths.size())).get(), neg_log_likelihood->get(), log_alpha->get(), blank.get(), reinterpret_cast<void*>(&zero_infinity));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_embed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset, nullable<int64_t> dim1, nullable<int64_t> dim2) {
  auto r_out = lantern_diag_embed_tensor_intt_intt_intt(self->get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagflat_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset) {
  auto r_out = lantern_diagflat_tensor_intt(self->get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> offset, nullable<int64_t> dim1, nullable<int64_t> dim2) {
  auto r_out = lantern_diagonal_tensor_intt_intt_intt(self->get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_self_Tensor_outdim_Dimname_dim1_Dimname_dim2_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> outdim, Rcpp::XPtr<XPtrTorchDimname> dim1, Rcpp::XPtr<XPtrTorchDimname> dim2, nullable<int64_t> offset) {
  auto r_out = lantern_diagonal_tensor_dimname_dimname_dimname_intt(self->get(), outdim->get(), dim1->get(), dim2->get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_backward_grad_Tensor_input_sizes_IntArrayRef_offset_int64_t_dim1_int64_t_dim2_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> input_sizes, nullable<int64_t> offset, nullable<int64_t> dim1, nullable<int64_t> dim2) {
  auto r_out = lantern_diagonal_backward_tensor_intarrayref_intt_intt_intt(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_sizes.data(), input_sizes.size())).get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_div_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_div_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_div_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_divide_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_true_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_true_divide_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_true_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dot_self_Tensor_tensor_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor) {
  auto r_out = lantern_dot_tensor_tensor(self->get(), tensor->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dot_out_out_Tensor_self_Tensor_tensor_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor) {
  auto r_out = lantern_dot_out_tensor_tensor_tensor(out->get(), self->get(), tensor->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vdot_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_vdot_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vdot_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_vdot_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_einsum_equation_stdstring_tensors_TensorList (std::string equation, Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_einsum_stdstring_tensorlist(XPtrTorchstring(lantern_string_new(equation.c_str())).get(), tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_weight_Tensor_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto r_out = lantern_embedding_tensor_tensor_intt_bool_bool(weight->get(), indices->get(), padding_idx.get(), reinterpret_cast<void*>(&scale_grad_by_freq), reinterpret_cast<void*>(&sparse));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool_sparse_bool (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> num_weights, nullable<int64_t> padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto r_out = lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(grad->get(), indices->get(), num_weights.get(), padding_idx.get(), reinterpret_cast<void*>(&scale_grad_by_freq), reinterpret_cast<void*>(&sparse));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_dense_backward_grad_output_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> num_weights, nullable<int64_t> padding_idx, bool scale_grad_by_freq) {
  auto r_out = lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool(grad_output->get(), indices->get(), num_weights.get(), padding_idx.get(), reinterpret_cast<void*>(&scale_grad_by_freq));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_renorm__self_Tensor_indices_Tensor_max_norm_double_norm_type_double (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, double max_norm, double norm_type) {
  auto r_out = lantern_embedding_renorm__tensor_tensor_double_double(self->get(), indices->get(), XPtrTorch(lantern_double(max_norm)).get(), XPtrTorch(lantern_double(norm_type)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_sparse_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> indices, nullable<int64_t> num_weights, nullable<int64_t> padding_idx, bool scale_grad_by_freq) {
  auto r_out = lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(grad->get(), indices->get(), num_weights.get(), padding_idx.get(), reinterpret_cast<void*>(&scale_grad_by_freq));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__embedding_bag_forward_only_weight_Tensor_indices_Tensor_offsets_Tensor (Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, bool scale_grad_by_freq, nullable<int64_t> mode, bool sparse, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights, bool include_last_offset) {
  auto r_out = lantern__embedding_bag_forward_only_tensor_tensor_tensor_bool_intt_bool_tensor_bool(weight->get(), indices->get(), offsets->get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), reinterpret_cast<void*>(&sparse), per_sample_weights->get(), reinterpret_cast<void*>(&include_last_offset));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor (Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, bool scale_grad_by_freq, nullable<int64_t> mode, bool sparse, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights, bool include_last_offset) {
  auto r_out = lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(weight->get(), indices->get(), offsets->get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), reinterpret_cast<void*>(&sparse), per_sample_weights->get(), reinterpret_cast<void*>(&include_last_offset));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor (Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, bool scale_grad_by_freq, nullable<int64_t> mode, bool sparse, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights, bool include_last_offset) {
  auto r_out = lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(weight->get(), indices->get(), offsets->get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), reinterpret_cast<void*>(&sparse), per_sample_weights->get(), reinterpret_cast<void*>(&include_last_offset));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_sparse_bool_per_sample_weights_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, Rcpp::XPtr<XPtrTorchTensor> offset2bag, Rcpp::XPtr<XPtrTorchTensor> bag_size, Rcpp::XPtr<XPtrTorchTensor> maximum_indices, nullable<int64_t> num_weights, bool scale_grad_by_freq, nullable<int64_t> mode, bool sparse, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights) {
  auto r_out = lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor(grad->get(), indices->get(), offsets->get(), offset2bag->get(), bag_size->get(), maximum_indices->get(), num_weights.get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), reinterpret_cast<void*>(&sparse), per_sample_weights->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_sparse_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, Rcpp::XPtr<XPtrTorchTensor> offset2bag, Rcpp::XPtr<XPtrTorchTensor> bag_size, nullable<int64_t> num_weights, bool scale_grad_by_freq, nullable<int64_t> mode, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights) {
  auto r_out = lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(grad->get(), indices->get(), offsets->get(), offset2bag->get(), bag_size->get(), num_weights.get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), per_sample_weights->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_dense_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, Rcpp::XPtr<XPtrTorchTensor> offset2bag, Rcpp::XPtr<XPtrTorchTensor> bag_size, Rcpp::XPtr<XPtrTorchTensor> maximum_indices, nullable<int64_t> num_weights, bool scale_grad_by_freq, nullable<int64_t> mode, Rcpp::XPtr<XPtrTorchTensor> per_sample_weights) {
  auto r_out = lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(grad->get(), indices->get(), offsets->get(), offset2bag->get(), bag_size->get(), maximum_indices->get(), num_weights.get(), reinterpret_cast<void*>(&scale_grad_by_freq), mode.get(), per_sample_weights->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_per_sample_weights_backward_grad_Tensor_weight_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_mode_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> offsets, Rcpp::XPtr<XPtrTorchTensor> offset2bag, nullable<int64_t> mode) {
  auto r_out = lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt(grad->get(), weight->get(), indices->get(), offsets->get(), offset2bag->get(), mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_meta_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_empty_meta_intarrayref_tensoroptions_memoryformat(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), names->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_empty_intarrayref_tensoroptions_memoryformat(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__empty_affine_quantized_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options, double scale, nullable<int64_t> zero_point, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get(), XPtrTorch(lantern_double(scale)).get(), zero_point.get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__empty_per_channel_affine_quantized_size_IntArrayRef_scales_Tensor_zero_points_Tensor_axis_int64_t (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensor> scales, Rcpp::XPtr<XPtrTorchTensor> zero_points, nullable<int64_t> axis, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), scales->get(), zero_points->get(), axis.get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_quantized_size_IntArrayRef_qtensor_Tensor (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensor> qtensor) {
  auto r_out = lantern_empty_quantized_intarrayref_tensor(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), qtensor->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_empty_out_tensor_intarrayref_memoryformat(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_like_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_empty_like_tensor_tensoroptions_memoryformat(self->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_strided_size_IntArrayRef_stride_IntArrayRef (std::vector<int64_t> size, std::vector<int64_t> stride, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_empty_strided_intarrayref_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erf__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erf_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erfc_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erfc__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erfc_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp2_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp2__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_exp2_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_expm1_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_expm1__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_expm1_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_n_int64_t (nullable<int64_t> n, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_eye_intt_tensoroptions(n.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_n_int64_t_m_int64_t (nullable<int64_t> n, nullable<int64_t> m, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_eye_intt_intt_tensoroptions(n.get(), m.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_out_out_Tensor_n_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> n) {
  auto r_out = lantern_eye_out_tensor_intt(out->get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_out_out_Tensor_n_int64_t_m_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> n, nullable<int64_t> m) {
  auto r_out = lantern_eye_out_tensor_intt_intt(out->get(), n.get(), m.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> start_dim, nullable<int64_t> end_dim) {
  auto r_out = lantern_flatten_tensor_intt_intt(self->get(), start_dim.get(), end_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> start_dim, nullable<int64_t> end_dim, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_flatten_tensor_intt_intt_dimname(self->get(), start_dim.get(), end_dim.get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> start_dim, Rcpp::XPtr<XPtrTorchDimname> end_dim, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_flatten_tensor_dimname_dimname_dimname(self->get(), start_dim->get(), end_dim->get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dims, Rcpp::XPtr<XPtrTorchDimname> out_dim) {
  auto r_out = lantern_flatten_tensor_dimnamelist_dimname(self->get(), dims->get(), out_dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fill__self_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_fill__tensor_scalar(self->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fill__self_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_fill__tensor_tensor(self->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_floor_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_floor__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_floor_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_floor_divide_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_floor_divide_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_floor_divide_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_frac_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_frac__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_frac_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchScalar> fill_value, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), fill_value->get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchScalar> fill_value, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_full_intarrayref_scalar_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), fill_value->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_out_out_Tensor_size_IntArrayRef_fill_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchScalar> fill_value) {
  auto r_out = lantern_full_out_tensor_intarrayref_scalar(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), fill_value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_like_self_Tensor_fill_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> fill_value, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_full_like_tensor_scalar_tensoroptions_memoryformat(self->get(), fill_value->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_from_file_filename_stdstring (std::string filename, bool shared, nullable<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_from_file_stdstring_bool_intt_tensoroptions(XPtrTorchstring(lantern_string_new(filename.c_str())).get(), reinterpret_cast<void*>(&shared), size.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_gcd_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_gcd_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_gcd__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_lcm_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_lcm_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm__self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_lcm__tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern_grid_sampler_tensor_tensor_intt_intt_bool(input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_2d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool(input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_2d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool(grad_output->get(), input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__grid_sampler_2d_cpu_fallback_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern__grid_sampler_2d_cpu_fallback_tensor_tensor_intt_intt_bool(input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__grid_sampler_2d_cpu_fallback_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern__grid_sampler_2d_cpu_fallback_backward_tensor_tensor_tensor_intt_intt_bool(grad_output->get(), input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_3d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool(input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_3d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grid, nullable<int64_t> interpolation_mode, nullable<int64_t> padding_mode, bool align_corners) {
  auto r_out = lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool(grad_output->get(), input->get(), grid->get(), interpolation_mode.get(), padding_mode.get(), reinterpret_cast<void*>(&align_corners));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hann_window_window_length_int64_t (nullable<int64_t> window_length, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hann_window_intt_tensoroptions(window_length.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hann_window_window_length_int64_t_periodic_bool (nullable<int64_t> window_length, bool periodic, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hann_window_intt_bool_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t (nullable<int64_t> window_length, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hamming_window_intt_tensoroptions(window_length.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool (nullable<int64_t> window_length, bool periodic, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hamming_window_intt_bool_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double (nullable<int64_t> window_length, bool periodic, double alpha, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hamming_window_intt_bool_double_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), XPtrTorch(lantern_double(alpha)).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double_beta_double (nullable<int64_t> window_length, bool periodic, double alpha, double beta, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_hamming_window_intt_bool_double_double_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), XPtrTorch(lantern_double(alpha)).get(), XPtrTorch(lantern_double(beta)).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t (nullable<int64_t> window_length, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_kaiser_window_intt_tensoroptions(window_length.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t_periodic_bool (nullable<int64_t> window_length, bool periodic, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_kaiser_window_intt_bool_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t_periodic_bool_beta_double (nullable<int64_t> window_length, bool periodic, double beta, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_kaiser_window_intt_bool_double_tensoroptions(window_length.get(), reinterpret_cast<void*>(&periodic), XPtrTorch(lantern_double(beta)).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hinge_embedding_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, double margin, nullable<int64_t> reduction) {
  auto r_out = lantern_hinge_embedding_loss_tensor_tensor_double_intt(self->get(), target->get(), XPtrTorch(lantern_double(margin)).get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_group_norm_input_Tensor_num_groups_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, nullable<int64_t> num_groups, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, double eps, bool cudnn_enabled) {
  auto r_out = lantern_group_norm_tensor_intt_tensor_tensor_double_bool(input->get(), num_groups.get(), weight->get(), bias->get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&cudnn_enabled));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_group_norm_input_Tensor_weight_Tensor_bias_Tensor_FALSE_int64_t_C_int64_t_HxW_int64_t_group_int64_t_eps_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, nullable<int64_t> False, nullable<int64_t> C, nullable<int64_t> HxW, nullable<int64_t> group, double eps) {
  auto r_out = lantern_native_group_norm_tensor_tensor_tensor_intt_intt_intt_intt_double(input->get(), weight->get(), bias->get(), False.get(), C.get(), HxW.get(), group.get(), XPtrTorch(lantern_double(eps)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_group_norm_backward_grad_out_Tensor_input_Tensor_mean_Tensor_rstd_Tensor_weight_Tensor_FALSE_int64_t_C_int64_t_HxW_int64_t_group_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> rstd, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> False, nullable<int64_t> C, nullable<int64_t> HxW, nullable<int64_t> group, std::vector<bool> output_mask) {
  auto r_out = lantern_native_group_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_stdarraybool(grad_out->get(), input->get(), mean->get(), rstd->get(), weight->get(), False.get(), C.get(), HxW.get(), group.get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ifft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized) {
  auto r_out = lantern_ifft_tensor_intt_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized, bool onesided) {
  auto r_out = lantern_rfft_tensor_intt_bool_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_irfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized, bool onesided, std::vector<int64_t> signal_sizes) {
  auto r_out = lantern_irfft_tensor_intt_bool_bool_intarrayref(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), XPtrTorchvector_int64_t(lantern_vector_int64_t(signal_sizes.data(), signal_sizes.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_with_size_self_Tensor_signal_ndim_int64_t_complex_input_bool_complex_output_bool_inverse_bool_checked_signal_sizes_IntArrayRef_normalized_bool_onesided_bool_output_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool complex_input, bool complex_output, bool inverse, std::vector<int64_t> checked_signal_sizes, bool normalized, bool onesided, std::vector<int64_t> output_sizes) {
  auto r_out = lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_bool_bool_intarrayref(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&complex_input), reinterpret_cast<void*>(&complex_output), reinterpret_cast<void*>(&inverse), XPtrTorchvector_int64_t(lantern_vector_int64_t(checked_signal_sizes.data(), checked_signal_sizes.size())).get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_sizes.data(), output_sizes.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_with_size_self_Tensor_signal_ndim_int64_t_complex_input_bool_complex_output_bool_inverse_bool_checked_signal_sizes_IntArrayRef_normalization_int64_t_onesided_bool_output_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool complex_input, bool complex_output, bool inverse, std::vector<int64_t> checked_signal_sizes, nullable<int64_t> normalization, bool onesided, std::vector<int64_t> output_sizes) {
  auto r_out = lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_intt_bool_intarrayref(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&complex_input), reinterpret_cast<void*>(&complex_output), reinterpret_cast<void*>(&inverse), XPtrTorchvector_int64_t(lantern_vector_int64_t(checked_signal_sizes.data(), checked_signal_sizes.size())).get(), normalization.get(), reinterpret_cast<void*>(&onesided), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_sizes.data(), output_sizes.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__cufft_get_plan_cache_size_device_index_int64_t (nullable<int64_t> device_index) {
  auto r_out = lantern__cufft_get_plan_cache_size_intt(device_index.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__cufft_get_plan_cache_max_size_device_index_int64_t (nullable<int64_t> device_index) {
  auto r_out = lantern__cufft_get_plan_cache_max_size_intt(device_index.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_set_plan_cache_max_size_device_index_int64_t_max_size_int64_t (nullable<int64_t> device_index, nullable<int64_t> max_size) {
  lantern__cufft_set_plan_cache_max_size_intt_intt(device_index.get(), max_size.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_clear_plan_cache_device_index_int64_t (nullable<int64_t> device_index) {
  lantern__cufft_clear_plan_cache_intt(device_index.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_self_Tensor_indices_TensorList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices) {
  auto r_out = lantern_index_tensor_tensorlist(self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_index_copy_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_index_copy_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_put__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices, Rcpp::XPtr<XPtrTorchTensor> values, bool accumulate) {
  auto r_out = lantern_index_put__tensor_tensorlist_tensor_bool(self->get(), indices->get(), values->get(), reinterpret_cast<void*>(&accumulate));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_put_self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices, Rcpp::XPtr<XPtrTorchTensor> values, bool accumulate) {
  auto r_out = lantern_index_put_tensor_tensorlist_tensor_bool(self->get(), indices->get(), values->get(), reinterpret_cast<void*>(&accumulate));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__index_put_impl__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorList> indices, Rcpp::XPtr<XPtrTorchTensor> values, bool accumulate, bool unsafe) {
  auto r_out = lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool(self->get(), indices->get(), values->get(), reinterpret_cast<void*>(&accumulate), reinterpret_cast<void*>(&unsafe));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_instance_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_use_input_stats_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&use_input_stats), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&cudnn_enabled));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_inverse_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inverse_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_inverse_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__inverse_helper_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__inverse_helper_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isclose_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = lantern_isclose_tensor_tensor_double_double_bool(self->get(), other->get(), XPtrTorch(lantern_double(rtol)).get(), XPtrTorch(lantern_double(atol)).get(), reinterpret_cast<void*>(&equal_nan));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isnan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isnan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_distributed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_is_distributed_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_floating_point_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_is_floating_point_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_complex_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_is_complex_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isreal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isreal_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_is_nonzero_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_same_size_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_is_same_size_tensor_tensor(self->get(), other->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_signed_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_is_signed_tensor(self->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kl_div_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, bool log_target) {
  auto r_out = lantern_kl_div_tensor_tensor_intt_bool(self->get(), target->get(), reduction.get(), reinterpret_cast<void*>(&log_target));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kl_div_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, bool log_target) {
  auto r_out = lantern_kl_div_backward_tensor_tensor_tensor_intt_bool(grad_output->get(), self->get(), target->get(), reduction.get(), reinterpret_cast<void*>(&log_target));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_kthvalue_tensor_intt_intt_bool(self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool(values->get(), indices->get(), self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_kthvalue_tensor_intt_dimname_bool(self->get(), k.get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool(values->get(), indices->get(), self->get(), k.get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_layer_norm_input_Tensor_normalized_shape_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> input, std::vector<int64_t> normalized_shape, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, double eps, bool cudnn_enable) {
  auto r_out = lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(input->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(normalized_shape.data(), normalized_shape.size())).get(), weight->get(), bias->get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&cudnn_enable));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_input_Tensor_weight_Tensor_bias_Tensor_M_int64_t_FALSE_int64_t_eps_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, nullable<int64_t> M, nullable<int64_t> False, double eps) {
  auto r_out = lantern_native_layer_norm_tensor_tensor_tensor_intt_intt_double(input->get(), weight->get(), bias->get(), M.get(), False.get(), XPtrTorch(lantern_double(eps)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_backward_grad_out_Tensor_input_Tensor_mean_Tensor_rstd_Tensor_weight_Tensor_M_int64_t_FALSE_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> rstd, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> M, nullable<int64_t> False, std::vector<bool> output_mask) {
  auto r_out = lantern_native_layer_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool(grad_out->get(), input->get(), mean->get(), rstd->get(), weight->get(), M.get(), False.get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linear_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_linear_tensor_tensor_tensor(input->get(), weight->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_linear_input_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_mkldnn_linear_tensor_tensor_tensor(input->get(), weight->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_int8_weight_fp32_activation_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> packed, Rcpp::XPtr<XPtrTorchTensor> col_offsets, Rcpp::XPtr<XPtrTorchScalar> weight_scale, Rcpp::XPtr<XPtrTorchScalar> weight_zero_point, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(input->get(), weight->get(), packed->get(), col_offsets->get(), weight_scale->get(), weight_zero_point->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_int8_weight_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> packed, Rcpp::XPtr<XPtrTorchTensor> col_offsets, Rcpp::XPtr<XPtrTorchScalar> weight_scale, Rcpp::XPtr<XPtrTorchScalar> weight_zero_point, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(input->get(), weight->get(), packed->get(), col_offsets->get(), weight_scale->get(), weight_zero_point->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fbgemm_linear_quantize_weight_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> input) {
  auto r_out = lantern_fbgemm_linear_quantize_weight_tensor(input->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),reinterpret_and_clean<double, lantern_double_delete>(lantern_vector_get(r_out, 2)),reinterpret_and_clean<int64_t, lantern_int64_t_delete>(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_gemm_matrix_fp16_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> input) {
  auto r_out = lantern_fbgemm_pack_gemm_matrix_fp16_tensor(input->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_fp16_weight_fp32_activation_input_Tensor_packed_weight_Tensor_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> packed_weight, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(input->get(), packed_weight->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_fp16_weight_input_Tensor_packed_weight_Tensor_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> packed_weight, Rcpp::XPtr<XPtrTorchTensor> bias) {
  auto r_out = lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(input->get(), packed_weight->get(), bias->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> input) {
  auto r_out = lantern_fbgemm_pack_quantized_matrix_tensor(input->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor_K_int64_t_FALSE_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, nullable<int64_t> K, nullable<int64_t> False) {
  auto r_out = lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(input->get(), K.get(), False.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linspace_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, nullable<int64_t> steps, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_linspace_scalar_scalar_intt_tensoroptions(start->get(), end->get(), steps.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linspace_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, nullable<int64_t> steps) {
  auto r_out = lantern_linspace_out_tensor_scalar_scalar_intt(out->get(), start->get(), end->get(), steps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log10_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log10__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log10_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log1p_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log1p__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log1p_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log2_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log2__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log2_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logaddexp_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logaddexp_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp2_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logaddexp2_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp2_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_logaddexp2_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logdet_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_logdet_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logspace_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, nullable<int64_t> steps, double base, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_logspace_scalar_scalar_intt_double_tensoroptions(start->get(), end->get(), steps.get(), XPtrTorch(lantern_double(base)).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logspace_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, nullable<int64_t> steps, double base) {
  auto r_out = lantern_logspace_out_tensor_scalar_scalar_intt_double(out->get(), start->get(), end->get(), steps.get(), XPtrTorch(lantern_double(base)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_log_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_log_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool half_to_float) {
  auto r_out = lantern__log_softmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&half_to_float));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__log_softmax_backward_data_tensor_tensor_intt_tensor(grad_output->get(), output->get(), dim.get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__logcumsumexp_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__logcumsumexp_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__logcumsumexp_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__logcumsumexp_out_tensor_tensor_intt(out->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_logcumsumexp_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_logcumsumexp_out_tensor_tensor_intt(out->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_logcumsumexp_tensor_dimname(self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_logcumsumexp_out_tensor_tensor_dimname(out->get(), self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_logsumexp_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_logsumexp_out_tensor_tensor_intarrayref_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_logsumexp_tensor_dimnamelist_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_margin_ranking_loss_input1_Tensor_input2_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> input1, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> target, double margin, nullable<int64_t> reduction) {
  auto r_out = lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(input1->get(), input2->get(), target->get(), XPtrTorch(lantern_double(margin)).get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matmul_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_matmul_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matmul_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_matmul_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_rank_self_Tensor_tol_double (Rcpp::XPtr<XPtrTorchTensor> self, double tol, bool symmetric) {
  auto r_out = lantern_matrix_rank_tensor_double_bool(self->get(), XPtrTorch(lantern_double(tol)).get(), reinterpret_cast<void*>(&symmetric));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_rank_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool symmetric) {
  auto r_out = lantern_matrix_rank_tensor_bool(self->get(), reinterpret_cast<void*>(&symmetric));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_power_self_Tensor_n_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n) {
  auto r_out = lantern_matrix_power_tensor_intt(self->get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_exp_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_matrix_exp_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_exp_backward_self_Tensor_grad_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad) {
  auto r_out = lantern_matrix_exp_backward_tensor_tensor(self->get(), grad->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__aminmax_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__aminmax_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__aminmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern__aminmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__compute_linear_combination_input_Tensor_coefficients_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> coefficients) {
  auto r_out = lantern__compute_linear_combination_tensor_tensor(input->get(), coefficients->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__compute_linear_combination_out_out_Tensor_input_Tensor_coefficients_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> coefficients) {
  auto r_out = lantern__compute_linear_combination_out_tensor_tensor_tensor(out->get(), input->get(), coefficients->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_max_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> max, Rcpp::XPtr<XPtrTorchTensor> max_values, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_max_out_tensor_tensor_tensor_intt_bool(max->get(), max_values->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_max_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> max, Rcpp::XPtr<XPtrTorchTensor> max_values, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_max_out_tensor_tensor_tensor_dimname_bool(max->get(), max_values->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_value_selecting_reduction_backward_grad_Tensor_dim_int64_t_indices_Tensor_sizes_IntArrayRef_keepdim_bool (Rcpp::XPtr<XPtrTorchTensor> grad, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> sizes, bool keepdim) {
  auto r_out = lantern_value_selecting_reduction_backward_tensor_intt_tensor_intarrayref_bool(grad->get(), dim.get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(sizes.data(), sizes.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amax_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_amax_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amax_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_amax_out_tensor_tensor_intarrayref_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool1d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool1d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool3d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_max_pool1d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_quantized_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_mean_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_mean_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_mean_tensor_dimnamelist_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_median_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_median_out_tensor_tensor_tensor_intt_bool(values->get(), indices->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_median_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_median_out_tensor_tensor_tensor_dimname_bool(values->get(), indices->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_min_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> min, Rcpp::XPtr<XPtrTorchTensor> min_indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_min_out_tensor_tensor_tensor_intt_bool(min->get(), min_indices->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_min_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> min, Rcpp::XPtr<XPtrTorchTensor> min_indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_min_out_tensor_tensor_tensor_dimname_bool(min->get(), min_indices->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_amin_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amin_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_amin_out_tensor_tensor_intarrayref_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (std::vector<int64_t> self_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool bias_defined) {
  auto r_out = lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(self_size.data(), self_size.size())).get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&bias_defined));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_weights_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool bias_defined) {
  auto r_out = lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&bias_defined));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, std::vector<bool> output_mask) {
  auto r_out = lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double exponential_average_factor, double epsilon) {
  auto r_out = lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(exponential_average_factor)).get(), XPtrTorch(lantern_double(epsilon)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, Rcpp::XPtr<XPtrTorchTensor> save_mean, Rcpp::XPtr<XPtrTorchTensor> save_var, double epsilon) {
  auto r_out = lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(input->get(), grad_output->get(), weight->get(), running_mean->get(), running_var->get(), save_mean->get(), save_var->get(), XPtrTorch(lantern_double(epsilon)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> self_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(self_size.data(), self_size.size())).get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_bias_grad_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output) {
  auto r_out = lantern_miopen_convolution_backward_bias_tensor(grad_output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> self_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(self_size.data(), self_size.size())).get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_depthwise_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups, bool benchmark, bool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(XPtrTorchvector_int64_t(lantern_vector_int64_t(weight_size.data(), weight_size.size())).get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get(), reinterpret_cast<void*>(&benchmark), reinterpret_cast<void*>(&deterministic));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> weight, nullable<int64_t> weight_stride0, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> cx, nullable<int64_t> mode, nullable<int64_t> hidden_size, nullable<int64_t> num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> dropout_state) {
  auto r_out = lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(input->get(), weight->get(), weight_stride0.get(), hx->get(), cx->get(), mode.get(), hidden_size.get(), num_layers.get(), reinterpret_cast<void*>(&batch_first), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), XPtrTorchvector_int64_t(lantern_vector_int64_t(batch_sizes.data(), batch_sizes.size())).get(), dropout_state->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> weight, nullable<int64_t> weight_stride0, Rcpp::XPtr<XPtrTorchTensor> weight_buf, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> cx, Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> grad_cy, nullable<int64_t> mode, nullable<int64_t> hidden_size, nullable<int64_t> num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> dropout_state, Rcpp::XPtr<XPtrTorchTensor> reserve, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(input->get(), weight->get(), weight_stride0.get(), weight_buf->get(), hx->get(), cx->get(), output->get(), grad_output->get(), grad_hy->get(), grad_cy->get(), mode.get(), hidden_size.get(), num_layers.get(), reinterpret_cast<void*>(&batch_first), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), XPtrTorchvector_int64_t(lantern_vector_int64_t(batch_sizes.data(), batch_sizes.size())).get(), dropout_state->get(), reserve->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),make_xptr<XPtrTorchTensorList>(lantern_vector_get(r_out, 3), "TensorList"));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_mm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mm_out_out_Tensor_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_mm_out_tensor_tensor_tensor(out->get(), self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_mm_sparse_Tensor_dense_Tensor (Rcpp::XPtr<XPtrTorchTensor> sparse, Rcpp::XPtr<XPtrTorchTensor> dense) {
  auto r_out = lantern__sparse_mm_tensor_tensor(sparse->get(), dense->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_mode_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_mode_out_tensor_tensor_tensor_intt_bool(values->get(), indices->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_mode_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim) {
  auto r_out = lantern_mode_out_tensor_tensor_tensor_dimname_bool(values->get(), indices->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_mul_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_mul_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_mul_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_multiply_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_multiply_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_multiply_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mv_self_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec) {
  auto r_out = lantern_mv_tensor_tensor(self->get(), vec->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mv_out_out_Tensor_self_Tensor_vec_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec) {
  auto r_out = lantern_mv_out_tensor_tensor_tensor(out->get(), self->get(), vec->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mvlgamma_self_Tensor_p_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> p) {
  auto r_out = lantern_mvlgamma_tensor_intt(self->get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> length) {
  auto r_out = lantern_narrow_tensor_intt_intt_intt(self->get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_self_Tensor_dim_int64_t_start_Tensor_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> start, nullable<int64_t> length) {
  auto r_out = lantern_narrow_tensor_intt_tensor_intt(self->get(), dim.get(), start->get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double momentum, double eps) {
  auto r_out = lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_out_out_Tensor_save_mean_Tensor_save_invstd_Tensor_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> save_mean, Rcpp::XPtr<XPtrTorchTensor> save_invstd, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, bool training, double momentum, double eps) {
  auto r_out = lantern_native_batch_norm_out_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_double(out->get(), save_mean->get(), save_invstd->get(), input->get(), weight->get(), bias->get(), running_mean->get(), running_var->get(), reinterpret_cast<void*>(&training), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_stats_input_Tensor_eps_double (Rcpp::XPtr<XPtrTorchTensor> input, double eps) {
  auto r_out = lantern_batch_norm_stats_tensor_double(input->get(), XPtrTorch(lantern_double(eps)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_elemt_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, double eps) {
  auto r_out = lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double(input->get(), weight->get(), bias->get(), mean->get(), invstd->get(), XPtrTorch(lantern_double(eps)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_elemt_out_out_Tensor_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, double eps) {
  auto r_out = lantern_batch_norm_elemt_out_tensor_tensor_tensor_tensor_tensor_tensor_double(out->get(), input->get(), weight->get(), bias->get(), mean->get(), invstd->get(), XPtrTorch(lantern_double(eps)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_count_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, double momentum, double eps, nullable<int64_t> count) {
  auto r_out = lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt(input->get(), mean->get(), invstd->get(), running_mean->get(), running_var->get(), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get(), count.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_with_counts_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_counts_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, double momentum, double eps, Rcpp::XPtr<XPtrTorchTensor> counts) {
  auto r_out = lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_tensor(input->get(), mean->get(), invstd->get(), running_mean->get(), running_var->get(), XPtrTorch(lantern_double(momentum)).get(), XPtrTorch(lantern_double(eps)).get(), counts->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_backward_grad_out_Tensor_input_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_invstd_Tensor_train_bool_eps_double_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, Rcpp::XPtr<XPtrTorchTensor> save_mean, Rcpp::XPtr<XPtrTorchTensor> save_invstd, bool train, double eps, std::vector<bool> output_mask) {
  auto r_out = lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(grad_out->get(), input->get(), weight->get(), running_mean->get(), running_var->get(), save_mean->get(), save_invstd->get(), reinterpret_cast<void*>(&train), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_backward_reduce_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_input_g_bool_weight_g_bool_bias_g_bool (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, Rcpp::XPtr<XPtrTorchTensor> weight, bool input_g, bool weight_g, bool bias_g) {
  auto r_out = lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool(grad_out->get(), input->get(), mean->get(), invstd->get(), weight->get(), reinterpret_cast<void*>(&input_g), reinterpret_cast<void*>(&weight_g), reinterpret_cast<void*>(&bias_g));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_backward_elemt_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_mean_dy_Tensor_mean_dy_xmu_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> invstd, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> mean_dy, Rcpp::XPtr<XPtrTorchTensor> mean_dy_xmu) {
  auto r_out = lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor(grad_out->get(), input->get(), mean->get(), invstd->get(), weight->get(), mean_dy->get(), mean_dy_xmu->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_update_stats_input_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> running_mean, Rcpp::XPtr<XPtrTorchTensor> running_var, double momentum) {
  auto r_out = lantern_batch_norm_update_stats_tensor_tensor_tensor_double(input->get(), running_mean->get(), running_var->get(), XPtrTorch(lantern_double(momentum)).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_input_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> weight, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref_intarrayref(input->get(), weight->get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__nnpack_spatial_convolution_backward_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding, std::vector<bool> output_mask) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool(input->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_backward_input_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> padding) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(input->get(), grad_output->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_backward_weight_input_Tensor_weightsize_IntArrayRef_grad_output_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> input, std::vector<int64_t> weightsize, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> padding) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(input->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(weightsize.data(), weightsize.size())).get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_ones_intarrayref_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_ones_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size) {
  auto r_out = lantern_ones_out_tensor_intarrayref(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_like_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_ones_like_tensor_tensoroptions_memoryformat(self->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pairwise_distance_x1_Tensor_x2_Tensor (Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2, double p, double eps, bool keepdim) {
  auto r_out = lantern_pairwise_distance_tensor_tensor_double_double_bool(x1->get(), x2->get(), XPtrTorch(lantern_double(p)).get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cdist_x1_Tensor_x2_Tensor (Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2, double p, nullable<int64_t> compute_mode) {
  auto r_out = lantern_cdist_tensor_tensor_double_intt(x1->get(), x2->get(), XPtrTorch(lantern_double(p)).get(), compute_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__euclidean_dist_x1_Tensor_x2_Tensor (Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2) {
  auto r_out = lantern__euclidean_dist_tensor_tensor(x1->get(), x2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cdist_forward_x1_Tensor_x2_Tensor_p_double_compute_mode_int64_t (Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2, double p, nullable<int64_t> compute_mode) {
  auto r_out = lantern__cdist_forward_tensor_tensor_double_intt(x1->get(), x2->get(), XPtrTorch(lantern_double(p)).get(), compute_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cdist_backward_grad_Tensor_x1_Tensor_x2_Tensor_p_double_cdist_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2, double p, Rcpp::XPtr<XPtrTorchTensor> cdist) {
  auto r_out = lantern__cdist_backward_tensor_tensor_tensor_double_tensor(grad->get(), x1->get(), x2->get(), XPtrTorch(lantern_double(p)).get(), cdist->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pdist_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double p) {
  auto r_out = lantern_pdist_tensor_double(self->get(), XPtrTorch(lantern_double(p)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pdist_forward_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double p) {
  auto r_out = lantern__pdist_forward_tensor_double(self->get(), XPtrTorch(lantern_double(p)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pdist_backward_grad_Tensor_self_Tensor_p_double_pdist_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, double p, Rcpp::XPtr<XPtrTorchTensor> pdist) {
  auto r_out = lantern__pdist_backward_tensor_tensor_double_tensor(grad->get(), self->get(), XPtrTorch(lantern_double(p)).get(), pdist->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosine_similarity_x1_Tensor_x2_Tensor (Rcpp::XPtr<XPtrTorchTensor> x1, Rcpp::XPtr<XPtrTorchTensor> x2, nullable<int64_t> dim, double eps) {
  auto r_out = lantern_cosine_similarity_tensor_tensor_intt_double(x1->get(), x2->get(), dim.get(), XPtrTorch(lantern_double(eps)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_movedim_self_Tensor_source_IntArrayRef_destination_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> source, std::vector<int64_t> destination) {
  auto r_out = lantern_movedim_tensor_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(source.data(), source.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(destination.data(), destination.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_movedim_self_Tensor_source_int64_t_destination_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> source, nullable<int64_t> destination) {
  auto r_out = lantern_movedim_tensor_intt_intt(self->get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pixel_shuffle_self_Tensor_upscale_factor_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> upscale_factor) {
  auto r_out = lantern_pixel_shuffle_tensor_intt(self->get(), upscale_factor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_channel_shuffle_self_Tensor_groups_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> groups) {
  auto r_out = lantern_channel_shuffle_tensor_intt(self->get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pinverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, double rcond) {
  auto r_out = lantern_pinverse_tensor_double(self->get(), XPtrTorch(lantern_double(rcond)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_poisson_nll_loss_input_Tensor_target_Tensor_log_input_bool_full_bool_eps_double_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> target, bool log_input, bool full, double eps, nullable<int64_t> reduction) {
  auto r_out = lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(input->get(), target->get(), reinterpret_cast<void*>(&log_input), reinterpret_cast<void*>(&full), XPtrTorch(lantern_double(eps)).get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rad2deg_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rad2deg__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rad2deg_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_deg2rad_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_deg2rad__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_deg2rad_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scalar_tensor_s_Scalar (Rcpp::XPtr<XPtrTorchScalar> s, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_scalar_tensor_scalar_tensoroptions(s->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_rand_intarrayref_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_rand_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_rand_intarrayref_generator_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size) {
  auto r_out = lantern_rand_out_tensor_intarrayref(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef_generator_Generator (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rand_out_tensor_intarrayref_generator(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_like_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_rand_like_tensor_tensoroptions_memoryformat(self->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef (nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randint_intt_intarrayref_tensoroptions(high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef_generator_Generator (nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randint_intt_intarrayref_generator_tensoroptions(high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef (nullable<int64_t> low, nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randint_intt_intt_intarrayref_tensoroptions(low.get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (nullable<int64_t> low, nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randint_intt_intt_intarrayref_generator_tensoroptions(low.get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> high, std::vector<int64_t> size) {
  auto r_out = lantern_randint_out_tensor_intt_intarrayref(out->get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef_generator_Generator (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_randint_out_tensor_intt_intarrayref_generator(out->get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> low, nullable<int64_t> high, std::vector<int64_t> size) {
  auto r_out = lantern_randint_out_tensor_intt_intt_intarrayref(out->get(), low.get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> low, nullable<int64_t> high, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_randint_out_tensor_intt_intt_intarrayref_generator(out->get(), low.get(), high.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_like_self_Tensor_high_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> high, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_randint_like_tensor_intt_tensoroptions_memoryformat(self->get(), high.get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_like_self_Tensor_low_int64_t_high_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> low, nullable<int64_t> high, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_randint_like_tensor_intt_intt_tensoroptions_memoryformat(self->get(), low.get(), high.get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randn_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randn_intarrayref_generator_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randn_intarrayref_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size) {
  auto r_out = lantern_randn_out_tensor_intarrayref(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef_generator_Generator (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_randn_out_tensor_intarrayref_generator(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_like_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_randn_like_tensor_tensoroptions_memoryformat(self->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_n_int64_t (nullable<int64_t> n, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randperm_intt_tensoroptions(n.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_n_int64_t_generator_Generator (nullable<int64_t> n, Rcpp::XPtr<XPtrTorch> generator, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_randperm_intt_generator_tensoroptions(n.get(), generator->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> n) {
  auto r_out = lantern_randperm_out_tensor_intt(out->get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t_generator_Generator (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> n, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_randperm_out_tensor_intt_generator(out->get(), n.get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_range_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchScalar> step, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_range_scalar_scalar_scalar_tensoroptions(start->get(), end->get(), step->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_range_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> start, Rcpp::XPtr<XPtrTorchScalar> end, Rcpp::XPtr<XPtrTorchScalar> step) {
  auto r_out = lantern_range_out_tensor_scalar_scalar_scalar(out->get(), start->get(), end->get(), step->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_reciprocal_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_reciprocal__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_reciprocal_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_neg_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_neg__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_neg_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_negative_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_negative__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_negative_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_repeats_Tensor (Rcpp::XPtr<XPtrTorchTensor> repeats) {
  auto r_out = lantern_repeat_interleave_tensor(repeats->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> repeats, nullable<int64_t> dim) {
  auto r_out = lantern_repeat_interleave_tensor_tensor_intt(self->get(), repeats->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> repeats, nullable<int64_t> dim) {
  auto r_out = lantern_repeat_interleave_tensor_intt_intt(self->get(), repeats.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> shape) {
  auto r_out = lantern_reshape_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(shape.data(), shape.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> shape) {
  auto r_out = lantern__mkldnn_reshape_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(shape.data(), shape.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_round_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_round__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_round_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rrelu_tensor_scalar_scalar_bool_generator(self->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rrelu__tensor_scalar_scalar_bool_generator(self->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_relu_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_relu__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prelu_self_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_prelu_tensor_tensor(self->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_prelu_backward_tensor_tensor_tensor(grad_output->get(), self->get(), weight->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_gelu_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_backward_grad_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_gelu_backward_tensor_tensor(grad->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_infinitely_differentiable_gelu_backward_grad_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_infinitely_differentiable_gelu_backward_tensor_tensor(grad->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_hardshrink_tensor_scalar(self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_hardshrink_backward_tensor_tensor_scalar(grad_out->get(), self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rsqrt_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rsqrt__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_rsqrt_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_self_Tensor_dim_Dimname_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, nullable<int64_t> index) {
  auto r_out = lantern_select_tensor_dimname_intt(self->get(), dim->get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_self_Tensor_dim_int64_t_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> index) {
  auto r_out = lantern_select_tensor_intt_intt(self->get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_backward_grad_Tensor_input_sizes_IntArrayRef_dim_int64_t_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> input_sizes, nullable<int64_t> dim, nullable<int64_t> index) {
  auto r_out = lantern_select_backward_tensor_intarrayref_intt_intt(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_sizes.data(), input_sizes.size())).get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_selu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_selu_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_selu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_selu__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_celu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_celu_tensor_scalar(self->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_celu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_celu__tensor_scalar(self->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_silu_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_silu__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_silu_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_silu_backward_tensor_tensor(grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sigmoid_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sigmoid__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sigmoid_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_logit_tensor_double(self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_logit__tensor_double(self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_logit_out_tensor_tensor_double(out->get(), self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sin_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sin__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sin_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sinh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sinh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sinh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_detach_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_detach_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_detach__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_detach__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_size_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_size_tensor_intt(self->get(), dim.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_size_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_size_tensor_dimname(self->get(), dim->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slice_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> end, nullable<int64_t> step) {
  auto r_out = lantern_slice_tensor_intt_intt_intt_intt(self->get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slice_backward_grad_Tensor_input_sizes_IntArrayRef_dim_int64_t_start_int64_t_end_int64_t_step_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> input_sizes, nullable<int64_t> dim, nullable<int64_t> start, nullable<int64_t> end, nullable<int64_t> step) {
  auto r_out = lantern_slice_backward_tensor_intarrayref_intt_intt_intt_intt(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_sizes.data(), input_sizes.size())).get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slogdet_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_slogdet_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smm_self_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_smm_tensor_tensor(self->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool half_to_float) {
  auto r_out = lantern__softmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&half_to_float));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__softmax_backward_data_tensor_tensor_intt_tensor(grad_output->get(), output->get(), dim.get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_unsafe_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> split_size, nullable<int64_t> dim) {
  auto r_out = lantern_unsafe_split_tensor_intt_intt(self->get(), split_size.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> split_size, nullable<int64_t> dim) {
  auto r_out = lantern_split_tensor_intt_intt(self->get(), split_size.get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_unsafe_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> split_sizes, nullable<int64_t> dim) {
  auto r_out = lantern_unsafe_split_with_sizes_tensor_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(split_sizes.data(), split_sizes.size())).get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> split_sizes, nullable<int64_t> dim) {
  auto r_out = lantern_split_with_sizes_tensor_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(split_sizes.data(), split_sizes.size())).get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_squeeze_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_squeeze_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_squeeze_tensor_dimname(self->get(), dim->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar(self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sspaddmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stack_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern_stack_tensorlist_intt(tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stack_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern_stack_out_tensor_tensorlist_intt(out->get(), tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hstack_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_hstack_tensorlist(tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hstack_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_hstack_out_tensor_tensorlist(out->get(), tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vstack_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_vstack_tensorlist(tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vstack_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_vstack_out_tensor_tensorlist(out->get(), tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dstack_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_dstack_tensorlist(tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dstack_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_dstack_out_tensor_tensorlist(out->get(), tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n_fft, nullable<int64_t> hop_length, nullable<int64_t> win_length, Rcpp::XPtr<XPtrTorchTensor> window, bool normalized, bool onesided, bool return_complex) {
  auto r_out = lantern_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(self->get(), n_fft.get(), hop_length.get(), win_length.get(), window->get(), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), reinterpret_cast<void*>(&return_complex));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_istft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n_fft, nullable<int64_t> hop_length, nullable<int64_t> win_length, Rcpp::XPtr<XPtrTorchTensor> window, bool center, bool normalized, bool onesided, nullable<int64_t> length, bool return_complex) {
  auto r_out = lantern_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(self->get(), n_fft.get(), hop_length.get(), win_length.get(), window->get(), reinterpret_cast<void*>(&center), reinterpret_cast<void*>(&normalized), reinterpret_cast<void*>(&onesided), length.get(), reinterpret_cast<void*>(&return_complex));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_stride_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_stride_tensor_intt(self->get(), dim.get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_stride_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_stride_tensor_dimname(self->get(), dim->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_sum_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_sum_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_sum_tensor_dimnamelist_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_nansum_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_nansum_tensor_intarrayref_bool_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_nansum_out_tensor_tensor_intarrayref_bool_scalartype(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sqrt_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sqrt__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sqrt_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_square_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_square_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_square__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_square__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_std_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_std_mean_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_mean_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_mean_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_intarrayref_bool_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_dimnamelist_bool_bool(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_prod_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_prod_tensor_intt_bool_scalartype(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_prod_out_tensor_tensor_intt_bool_scalartype(out->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_prod_tensor_dimname_bool_scalartype(self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_prod_out_tensor_tensor_dimname_bool_scalartype(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_t_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tan_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tan__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tan_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tanh_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tanh__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_tanh_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tensordot_self_Tensor_other_Tensor_dims_self_IntArrayRef_dims_other_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, std::vector<int64_t> dims_self, std::vector<int64_t> dims_other) {
  auto r_out = lantern_tensordot_tensor_tensor_intarrayref_intarrayref(self->get(), other->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims_self.data(), dims_self.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims_other.data(), dims_other.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> threshold, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_threshold_tensor_scalar_scalar(self->get(), threshold->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold__self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> threshold, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_threshold__tensor_scalar_scalar(self->get(), threshold->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_out_out_Tensor_self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> threshold, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_threshold_out_tensor_tensor_scalar_scalar(out->get(), self->get(), threshold->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_backward_grad_output_Tensor_self_Tensor_threshold_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> threshold) {
  auto r_out = lantern_threshold_backward_tensor_tensor_scalar(grad_output->get(), self->get(), threshold->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim0, nullable<int64_t> dim1) {
  auto r_out = lantern_transpose_tensor_intt_intt(self->get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim0, Rcpp::XPtr<XPtrTorchDimname> dim1) {
  auto r_out = lantern_transpose_tensor_dimname_dimname(self->get(), dim0->get(), dim1->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim0, nullable<int64_t> dim1) {
  auto r_out = lantern__mkldnn_transpose_tensor_intt_intt(self->get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim0, nullable<int64_t> dim1) {
  auto r_out = lantern__mkldnn_transpose__tensor_intt_intt(self->get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_one_hot_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> num_classes) {
  auto r_out = lantern_one_hot_tensor_intt(self->get(), num_classes.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flip_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dims) {
  auto r_out = lantern_flip_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fliplr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_fliplr_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flipud_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_flipud_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_roll_self_Tensor_shifts_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
  auto r_out = lantern_roll_tensor_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(shifts.data(), shifts.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rot90_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, std::vector<int64_t> dims) {
  auto r_out = lantern_rot90_tensor_intt_intarrayref(self->get(), k.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dims.data(), dims.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapz_y_Tensor_x_Tensor (Rcpp::XPtr<XPtrTorchTensor> y, Rcpp::XPtr<XPtrTorchTensor> x, nullable<int64_t> dim) {
  auto r_out = lantern_trapz_tensor_tensor_intt(y->get(), x->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapz_y_Tensor (Rcpp::XPtr<XPtrTorchTensor> y, double dx, nullable<int64_t> dim) {
  auto r_out = lantern_trapz_tensor_double_intt(y->get(), XPtrTorch(lantern_double(dx)).get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__trilinear_i1_Tensor_i2_Tensor_i3_Tensor_expand1_IntArrayRef_expand2_IntArrayRef_expand3_IntArrayRef_sumdim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> i1, Rcpp::XPtr<XPtrTorchTensor> i2, Rcpp::XPtr<XPtrTorchTensor> i3, std::vector<int64_t> expand1, std::vector<int64_t> expand2, std::vector<int64_t> expand3, std::vector<int64_t> sumdim, nullable<int64_t> unroll_dim) {
  auto r_out = lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(i1->get(), i2->get(), i3->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(expand1.data(), expand1.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(expand2.data(), expand2.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(expand3.data(), expand3.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(sumdim.data(), sumdim.size())).get(), unroll_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triplet_margin_loss_anchor_Tensor_positive_Tensor_negative_Tensor (Rcpp::XPtr<XPtrTorchTensor> anchor, Rcpp::XPtr<XPtrTorchTensor> positive, Rcpp::XPtr<XPtrTorchTensor> negative, double margin, double p, double eps, bool swap, nullable<int64_t> reduction) {
  auto r_out = lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(anchor->get(), positive->get(), negative->get(), XPtrTorch(lantern_double(margin)).get(), XPtrTorch(lantern_double(p)).get(), XPtrTorch(lantern_double(eps)).get(), reinterpret_cast<void*>(&swap), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_trunc_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_trunc__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_trunc_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_fix_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_fix__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_fix_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace__has_compatible_shallow_copy_type_self_Tensor_from_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> from) {
  auto r_out = lantern__has_compatible_shallow_copy_type_tensor_tensor(self->get(), from->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool sorted, bool return_inverse) {
  auto r_out = lantern__unique_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&sorted), reinterpret_cast<void*>(&return_inverse));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool sorted, bool return_inverse, bool return_counts) {
  auto r_out = lantern_unique_dim_tensor_intt_bool_bool_bool(self->get(), dim.get(), reinterpret_cast<void*>(&sorted), reinterpret_cast<void*>(&return_inverse), reinterpret_cast<void*>(&return_counts));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_consecutive_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool return_inverse, bool return_counts, nullable<int64_t> dim) {
  auto r_out = lantern_unique_consecutive_tensor_bool_bool_intt(self->get(), reinterpret_cast<void*>(&return_inverse), reinterpret_cast<void*>(&return_counts), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_consecutive_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool return_inverse, bool return_counts) {
  auto r_out = lantern_unique_dim_consecutive_tensor_intt_bool_bool(self->get(), dim.get(), reinterpret_cast<void*>(&return_inverse), reinterpret_cast<void*>(&return_counts));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique2_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool sorted, bool return_inverse, bool return_counts) {
  auto r_out = lantern__unique2_tensor_bool_bool_bool(self->get(), reinterpret_cast<void*>(&sorted), reinterpret_cast<void*>(&return_inverse), reinterpret_cast<void*>(&return_counts));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__unsafe_view_self_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> size) {
  auto r_out = lantern__unsafe_view_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_unsqueeze_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_unsqueeze_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vander_x_Tensor (Rcpp::XPtr<XPtrTorchTensor> x, nullable<int64_t> False, bool increasing) {
  auto r_out = lantern_vander_tensor_intt_bool(x->get(), False.get(), reinterpret_cast<void*>(&increasing));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_var_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_intarrayref_bool_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_dimnamelist_bool_bool(out->get(), self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern_var_mean_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_mean_tensor_intarrayref_bool_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dim, bool unbiased, bool keepdim) {
  auto r_out = lantern_var_mean_tensor_dimnamelist_bool_bool(self->get(), dim->get(), reinterpret_cast<void*>(&unbiased), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_where_tensor_tensor_tensor(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Scalar_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchScalar> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_where_tensor_scalar_tensor(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_where_tensor_tensor_scalar(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Scalar_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchScalar> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_where_tensor_scalar_scalar(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_where_condition_Tensor (Rcpp::XPtr<XPtrTorchTensor> condition) {
  auto r_out = lantern_where_tensor(condition->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__s_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> condition, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern__s_where_tensor_tensor_tensor(condition->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_except_dim_v_Tensor (Rcpp::XPtr<XPtrTorchTensor> v, nullable<int64_t> pow, nullable<int64_t> dim) {
  auto r_out = lantern_norm_except_dim_tensor_intt_intt(v->get(), pow.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__weight_norm_v_Tensor_g_Tensor (Rcpp::XPtr<XPtrTorchTensor> v, Rcpp::XPtr<XPtrTorchTensor> g, nullable<int64_t> dim) {
  auto r_out = lantern__weight_norm_tensor_tensor_intt(v->get(), g->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_v_Tensor_g_Tensor (Rcpp::XPtr<XPtrTorchTensor> v, Rcpp::XPtr<XPtrTorchTensor> g, nullable<int64_t> dim) {
  auto r_out = lantern__weight_norm_cuda_interface_tensor_tensor_intt(v->get(), g->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_w, Rcpp::XPtr<XPtrTorchTensor> saved_v, Rcpp::XPtr<XPtrTorchTensor> saved_g, Rcpp::XPtr<XPtrTorchTensor> saved_norms, nullable<int64_t> dim) {
  auto r_out = lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt(grad_w->get(), saved_v->get(), saved_g->get(), saved_norms->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_differentiable_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_w, Rcpp::XPtr<XPtrTorchTensor> saved_v, Rcpp::XPtr<XPtrTorchTensor> saved_g, Rcpp::XPtr<XPtrTorchTensor> saved_norms, nullable<int64_t> dim) {
  auto r_out = lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt(grad_w->get(), saved_v->get(), saved_g->get(), saved_norms->get(), dim.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> names, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_zeros_intarrayref_dimnamelist_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), names->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_zeros_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, std::vector<int64_t> size) {
  auto r_out = lantern_zeros_out_tensor_intarrayref(out->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_like_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensorOptions> options, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_zeros_like_tensor_tensoroptions_memoryformat(self->get(), options->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__standard_gamma_grad_self_Tensor_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern__standard_gamma_grad_tensor_tensor(self->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__standard_gamma_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern__standard_gamma_tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__dirichlet_grad_x_Tensor_alpha_Tensor_total_Tensor (Rcpp::XPtr<XPtrTorchTensor> x, Rcpp::XPtr<XPtrTorchTensor> alpha, Rcpp::XPtr<XPtrTorchTensor> total) {
  auto r_out = lantern__dirichlet_grad_tensor_tensor_tensor(x->get(), alpha->get(), total->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sample_dirichlet_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern__sample_dirichlet_tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_poisson_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_poisson_tensor_generator(self->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binomial_count_Tensor_prob_Tensor (Rcpp::XPtr<XPtrTorchTensor> count, Rcpp::XPtr<XPtrTorchTensor> prob, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_binomial_tensor_tensor_generator(count->get(), prob->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_native_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p) {
  auto r_out = lantern_native_norm_tensor_scalar(self->get(), p->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_native_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_native_norm_tensor_scalar_intarrayref_bool_scalartype(self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__sparse_sum_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_sum_tensor_scalartype(self->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim) {
  auto r_out = lantern__sparse_sum_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_sum_tensor_intarrayref_scalartype(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_backward_grad_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim) {
  auto r_out = lantern__sparse_sum_backward_tensor_tensor_intarrayref(grad->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool half_to_float) {
  auto r_out = lantern__sparse_softmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&half_to_float));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__sparse_softmax_backward_data_tensor_tensor_intt_tensor(grad_output->get(), output->get(), dim.get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_log_softmax_tensor_intt_scalartype(self->get(), dim.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern__sparse_log_softmax_tensor_dimname_scalartype(self->get(), dim->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool half_to_float) {
  auto r_out = lantern__sparse_log_softmax_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&half_to_float));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__sparse_log_softmax_backward_data_tensor_tensor_intt_tensor(grad_output->get(), output->get(), dim.get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_norm_tensor_scalar_scalartype(self->get(), p->get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p) {
  auto r_out = lantern_norm_tensor_scalar(self->get(), p->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_norm_tensor_scalar_intarrayref_bool_scalartype(self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_norm_tensor_scalar_intarrayref_bool(self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out->get(), self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(out->get(), self->get(), p->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_norm_tensor_scalar_dimnamelist_bool_scalartype(self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_norm_tensor_scalar_dimnamelist_bool(self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(out->get(), self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorch> dim, bool keepdim) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(out->get(), self->get(), p->get(), dim->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_frobenius_norm_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_frobenius_norm_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool keepdim) {
  auto r_out = lantern_nuclear_norm_tensor_bool(self->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, bool keepdim) {
  auto r_out = lantern_nuclear_norm_out_tensor_tensor_bool(out->get(), self->get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nuclear_norm_tensor_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dim.data(), dim.size())).get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clone_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_clone_tensor_memoryformat(self->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_resize_as__self_Tensor_the_template_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> the_template, Rcpp::XPtr<XPtrTorchMemoryFormat> memory_format) {
  auto r_out = lantern_resize_as__tensor_tensor_memoryformat(self->get(), the_template->get(), memory_format->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zero__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_zero__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_sub_out_tensor_tensor_tensor_scalar(out->get(), self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_sub_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_sub_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_subtract_out_tensor_tensor_tensor_scalar(out->get(), self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_subtract_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_subtract_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsub_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_rsub_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_heaviside_out_out_Tensor_self_Tensor_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values) {
  auto r_out = lantern_heaviside_out_tensor_tensor_tensor(out->get(), self->get(), values->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_heaviside_self_Tensor_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> values) {
  auto r_out = lantern_heaviside_tensor_tensor(self->get(), values->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsub_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_rsub_tensor_scalar_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_addmm_self_Tensor_sparse_Tensor_dense_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> sparse, Rcpp::XPtr<XPtrTorchTensor> dense, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(self->get(), sparse->get(), dense->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addmm_tensor_tensor_tensor_scalar_scalar(self->get(), mat1->get(), mat2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_size_IntArrayRef_options_TensorOptions (std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_sparse_coo_tensor_intarrayref_tensoroptions(XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor (Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(indices->get(), values->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> values, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(indices->get(), values->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_unsafe_indices_Tensor_values_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> values, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(indices->get(), values->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__validate_sparse_coo_tensor_args_indices_Tensor_values_Tensor_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> values, std::vector<int64_t> size) {
  lantern__validate_sparse_coo_tensor_args_tensor_tensor_intarrayref(indices->get(), values->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_with_dims_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_options_TensorOptions (nullable<int64_t> sparse_dim, nullable<int64_t> dense_dim, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions(sparse_dim.get(), dense_dim.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_with_dims_and_tensors_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_indices_Tensor_values_Tensor_options_TensorOptions (nullable<int64_t> sparse_dim, nullable<int64_t> dense_dim, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions(sparse_dim.get(), dense_dim.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), indices->get(), values->get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_to_dense_backward_grad_Tensor_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input) {
  auto r_out = lantern_to_dense_backward_tensor_tensor(grad->get(), input->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hspmm_out_out_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_hspmm_out_tensor_tensor_tensor(out->get(), mat1->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hspmm_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<XPtrTorchTensor> mat1, Rcpp::XPtr<XPtrTorchTensor> mat2) {
  auto r_out = lantern_hspmm_tensor_tensor(mat1->get(), mat2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copy_sparse_to_sparse__self_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> src, bool non_blocking) {
  auto r_out = lantern_copy_sparse_to_sparse__tensor_tensor_bool(self->get(), src->get(), reinterpret_cast<void*>(&non_blocking));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_unbind_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_unbind_tensor_intt(self->get(), dim.get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_unbind_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim) {
  auto r_out = lantern_unbind_tensor_dimname(self->get(), dim->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_reorder_conv2d_weight_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_reorder_conv3d_weight_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, nullable<int64_t> groups) {
  auto r_out = lantern_mkldnn_reorder_conv3d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_to_mkldnn_backward_grad_Tensor_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input) {
  auto r_out = lantern_to_mkldnn_backward_tensor_tensor(grad->get(), input->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantize_per_tensor_self_Tensor_scale_double_zero_point_int64_t_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, double scale, nullable<int64_t> zero_point, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_quantize_per_tensor_tensor_double_intt_scalartype(self->get(), XPtrTorch(lantern_double(scale)).get(), zero_point.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_quantize_per_tensor_tensors_TensorList_scales_Tensor_zero_points_Tensor_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchTensor> scales, Rcpp::XPtr<XPtrTorchTensor> zero_points, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_quantize_per_tensor_tensorlist_tensor_tensor_scalartype(tensors->get(), scales->get(), zero_points->get(), dtype->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantize_per_channel_self_Tensor_scales_Tensor_zero_points_Tensor_axis_int64_t_dtype_ScalarType (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scales, Rcpp::XPtr<XPtrTorchTensor> zero_points, nullable<int64_t> axis, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype(self->get(), scales->get(), zero_points->get(), axis.get(), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dequantize_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_dequantize_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_dequantize_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_dequantize_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
double cpp_torch_namespace_q_scale_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_q_scale_tensor(self->get());
return reinterpret_and_clean<double, lantern_double_delete>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_q_zero_point_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_q_zero_point_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_q_per_channel_scales_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_q_per_channel_scales_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_q_per_channel_zero_points_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_q_per_channel_zero_points_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_q_per_channel_axis_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_q_per_channel_axis_tensor(self->get());
return reinterpret_and_clean<int64_t, lantern_int64_t_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_int_repr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_int_repr_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__make_per_tensor_quantized_tensor_self_Tensor_scale_double_zero_point_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, double scale, nullable<int64_t> zero_point) {
  auto r_out = lantern__make_per_tensor_quantized_tensor_tensor_double_intt(self->get(), XPtrTorch(lantern_double(scale)).get(), zero_point.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__make_per_channel_quantized_tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> axis) {
  auto r_out = lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt(self->get(), scale->get(), zero_point->get(), axis.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_tensor_affine_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, double scale, nullable<int64_t> zero_point, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt(self->get(), XPtrTorch(lantern_double(scale)).get(), zero_point.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_tensor_affine_backward_grad_Tensor_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, double scale, nullable<int64_t> zero_point, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_backward_tensor_tensor_double_intt_intt_intt(grad->get(), self->get(), XPtrTorch(lantern_double(scale)).get(), zero_point.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fake_quantize_learnable_per_tensor_affine_self_Tensor_scale_Tensor_zero_point_Tensor_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern__fake_quantize_learnable_per_tensor_affine_tensor_tensor_tensor_intt_intt(self->get(), scale->get(), zero_point->get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fake_quantize_learnable_per_tensor_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern__fake_quantize_learnable_per_tensor_affine_backward_tensor_tensor_tensor_tensor_intt_intt(grad->get(), self->get(), scale->get(), zero_point->get(), quant_min.get(), quant_max.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_channel_affine_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> axis, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(self->get(), scale->get(), zero_point->get(), axis.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_channel_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> axis, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern_fake_quantize_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt(grad->get(), self->get(), scale->get(), zero_point->get(), axis.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fake_quantize_learnable_per_channel_affine_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> axis, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern__fake_quantize_learnable_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(self->get(), scale->get(), zero_point->get(), axis.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fake_quantize_learnable_per_channel_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> scale, Rcpp::XPtr<XPtrTorchTensor> zero_point, nullable<int64_t> axis, nullable<int64_t> quant_min, nullable<int64_t> quant_max) {
  auto r_out = lantern__fake_quantize_learnable_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt(grad->get(), self->get(), scale->get(), zero_point->get(), axis.get(), quant_min.get(), quant_max.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__choose_qparams_per_tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool reduce_range) {
  auto r_out = lantern__choose_qparams_per_tensor_tensor_bool(self->get(), reinterpret_cast<void*>(&reduce_range));
return Rcpp::List::create(reinterpret_and_clean<double, lantern_double_delete>(lantern_vector_get(r_out, 0)),reinterpret_and_clean<int64_t, lantern_int64_t_delete>(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__saturate_weight_to_fp16_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern__saturate_weight_to_fp16_tensor(weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_choose_qparams_optimized_input_Tensor_numel_int64_t_n_bins_int64_t_ratio_double_bit_width_int64_t (Rcpp::XPtr<XPtrTorchTensor> input, nullable<int64_t> numel, nullable<int64_t> n_bins, double ratio, nullable<int64_t> bit_width) {
  auto r_out = lantern_choose_qparams_optimized_tensor_intt_intt_double_intt(input->get(), numel.get(), n_bins.get(), XPtrTorch(lantern_double(ratio)).get(), bit_width.get());
return Rcpp::List::create(reinterpret_and_clean<double, lantern_double_delete>(lantern_vector_get(r_out, 0)),reinterpret_and_clean<double, lantern_double_delete>(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_meshgrid_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_meshgrid_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cartesian_prod_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern_cartesian_prod_tensorlist(tensors->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_combinations_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> r, bool with_replacement) {
  auto r_out = lantern_combinations_tensor_intt_bool(self->get(), r.get(), reinterpret_cast<void*>(&with_replacement));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_namespace_result_type_tensor_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> tensor, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_result_type_tensor_tensor(tensor->get(), other->get());
return make_xptr<XPtrTorchScalarType>(r_out, "ScalarType");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_namespace_result_type_tensor_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> tensor, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_result_type_tensor_scalar(tensor->get(), other->get());
return make_xptr<XPtrTorchScalarType>(r_out, "ScalarType");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_namespace_result_type_scalar_Scalar_tensor_Tensor (Rcpp::XPtr<XPtrTorchScalar> scalar, Rcpp::XPtr<XPtrTorchTensor> tensor) {
  auto r_out = lantern_result_type_scalar_tensor(scalar->get(), tensor->get());
return make_xptr<XPtrTorchScalarType>(r_out, "ScalarType");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_namespace_result_type_scalar1_Scalar_scalar2_Scalar (Rcpp::XPtr<XPtrTorchScalar> scalar1, Rcpp::XPtr<XPtrTorchScalar> scalar2) {
  auto r_out = lantern_result_type_scalar_scalar(scalar1->get(), scalar2->get());
return make_xptr<XPtrTorchScalarType>(r_out, "ScalarType");
}

// [[Rcpp::export]]
bool cpp_torch_namespace_can_cast_from_ScalarType_to_ScalarType (Rcpp::XPtr<XPtrTorch> from, Rcpp::XPtr<XPtrTorch> to) {
  auto r_out = lantern_can_cast_scalartype_scalartype(from->get(), to->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_namespace_promote_types_type1_ScalarType_type2_ScalarType (Rcpp::XPtr<XPtrTorch> type1, Rcpp::XPtr<XPtrTorch> type2) {
  auto r_out = lantern_promote_types_scalartype_scalartype(type1->get(), type2->get());
return make_xptr<XPtrTorchScalarType>(r_out, "ScalarType");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalar> cpp_torch_namespace__local_scalar_dense_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__local_scalar_dense_tensor(self->get());
return make_xptr<XPtrTorchScalar>(r_out, "Scalar");
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_input_gates_Tensor_hidden_gates_Tensor_cx_Tensor (Rcpp::XPtr<XPtrTorchTensor> input_gates, Rcpp::XPtr<XPtrTorchTensor> hidden_gates, Rcpp::XPtr<XPtrTorchTensor> cx, Rcpp::XPtr<XPtrTorchTensor> input_bias, Rcpp::XPtr<XPtrTorchTensor> hidden_bias) {
  auto r_out = lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor(input_gates->get(), hidden_gates->get(), cx->get(), input_bias->get(), hidden_bias->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_cx_Tensor_cy_Tensor_workspace_Tensor_has_bias_bool (Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> grad_cy, Rcpp::XPtr<XPtrTorchTensor> cx, Rcpp::XPtr<XPtrTorchTensor> cy, Rcpp::XPtr<XPtrTorchTensor> workspace, bool has_bias) {
  auto r_out = lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool(grad_hy->get(), grad_cy->get(), cx->get(), cy->get(), workspace->get(), reinterpret_cast<void*>(&has_bias));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_input_gates_Tensor_hidden_gates_Tensor_input_bias_Tensor_hidden_bias_Tensor_cx_Tensor_cy_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> grad_cy, Rcpp::XPtr<XPtrTorchTensor> input_gates, Rcpp::XPtr<XPtrTorchTensor> hidden_gates, Rcpp::XPtr<XPtrTorchTensor> input_bias, Rcpp::XPtr<XPtrTorchTensor> hidden_bias, Rcpp::XPtr<XPtrTorchTensor> cx, Rcpp::XPtr<XPtrTorchTensor> cy) {
  auto r_out = lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(grad_hy->get(), grad_cy->get(), input_gates->get(), hidden_gates->get(), input_bias->get(), hidden_bias->get(), cx->get(), cy->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor (Rcpp::XPtr<XPtrTorchTensor> input_gates, Rcpp::XPtr<XPtrTorchTensor> hidden_gates, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> input_bias, Rcpp::XPtr<XPtrTorchTensor> hidden_bias) {
  auto r_out = lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor(input_gates->get(), hidden_gates->get(), hx->get(), input_bias->get(), hidden_bias->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_backward_grad_hy_Tensor_workspace_Tensor_has_bias_bool (Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> workspace, bool has_bias) {
  auto r_out = lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool(grad_hy->get(), workspace->get(), reinterpret_cast<void*>(&has_bias));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_gru_cell_backward_grad_hy_Tensor_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor_input_bias_Tensor_hidden_bias_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_hy, Rcpp::XPtr<XPtrTorchTensor> input_gates, Rcpp::XPtr<XPtrTorchTensor> hidden_gates, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> input_bias, Rcpp::XPtr<XPtrTorchTensor> hidden_bias) {
  auto r_out = lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor(grad_hy->get(), input_gates->get(), hidden_gates->get(), hx->get(), input_bias->get(), hidden_bias->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)),XPtrTorchTensor(lantern_vector_get(r_out, 3)),XPtrTorchTensor(lantern_vector_get(r_out, 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_input_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool(input->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), reinterpret_cast<void*>(&batch_first));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_data_Tensor_batch_sizes_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> data, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, Rcpp::XPtr<XPtrTorchTensorList> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool(data->get(), batch_sizes->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), reinterpret_cast<void*>(&batch_first));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> data, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data->get(), batch_sizes->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), reinterpret_cast<void*>(&batch_first));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> data, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data->get(), batch_sizes->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional), reinterpret_cast<void*>(&batch_first));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<XPtrTorchTensor> data, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensorList> params, bool has_biases, nullable<int64_t> num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data->get(), batch_sizes->get(), hx->get(), params->get(), reinterpret_cast<void*>(&has_biases), num_layers.get(), XPtrTorch(lantern_double(dropout)).get(), reinterpret_cast<void*>(&train), reinterpret_cast<void*>(&bidirectional));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh) {
  auto r_out = lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh) {
  auto r_out = lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh) {
  auto r_out = lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh) {
  auto r_out = lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensorList> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh, Rcpp::XPtr<XPtrTorchTensor> packed_ih, Rcpp::XPtr<XPtrTorchTensor> packed_hh, Rcpp::XPtr<XPtrTorchTensor> col_offsets_ih, Rcpp::XPtr<XPtrTorchTensor> col_offsets_hh, Rcpp::XPtr<XPtrTorchScalar> scale_ih, Rcpp::XPtr<XPtrTorchScalar> scale_hh, Rcpp::XPtr<XPtrTorchScalar> zero_point_ih, Rcpp::XPtr<XPtrTorchScalar> zero_point_hh) {
  auto r_out = lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get(), packed_ih->get(), packed_hh->get(), col_offsets_ih->get(), col_offsets_hh->get(), scale_ih->get(), scale_hh->get(), zero_point_ih->get(), zero_point_hh->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh, Rcpp::XPtr<XPtrTorchTensor> packed_ih, Rcpp::XPtr<XPtrTorchTensor> packed_hh, Rcpp::XPtr<XPtrTorchTensor> col_offsets_ih, Rcpp::XPtr<XPtrTorchTensor> col_offsets_hh, Rcpp::XPtr<XPtrTorchScalar> scale_ih, Rcpp::XPtr<XPtrTorchScalar> scale_hh, Rcpp::XPtr<XPtrTorchScalar> zero_point_ih, Rcpp::XPtr<XPtrTorchScalar> zero_point_hh) {
  auto r_out = lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get(), packed_ih->get(), packed_hh->get(), col_offsets_ih->get(), col_offsets_hh->get(), scale_ih->get(), scale_hh->get(), zero_point_ih->get(), zero_point_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh, Rcpp::XPtr<XPtrTorchTensor> packed_ih, Rcpp::XPtr<XPtrTorchTensor> packed_hh, Rcpp::XPtr<XPtrTorchTensor> col_offsets_ih, Rcpp::XPtr<XPtrTorchTensor> col_offsets_hh, Rcpp::XPtr<XPtrTorchScalar> scale_ih, Rcpp::XPtr<XPtrTorchScalar> scale_hh, Rcpp::XPtr<XPtrTorchScalar> zero_point_ih, Rcpp::XPtr<XPtrTorchScalar> zero_point_hh) {
  auto r_out = lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get(), packed_ih->get(), packed_hh->get(), col_offsets_ih->get(), col_offsets_hh->get(), scale_ih->get(), scale_hh->get(), zero_point_ih->get(), zero_point_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> hx, Rcpp::XPtr<XPtrTorchTensor> w_ih, Rcpp::XPtr<XPtrTorchTensor> w_hh, Rcpp::XPtr<XPtrTorchTensor> b_ih, Rcpp::XPtr<XPtrTorchTensor> b_hh, Rcpp::XPtr<XPtrTorchTensor> packed_ih, Rcpp::XPtr<XPtrTorchTensor> packed_hh, Rcpp::XPtr<XPtrTorchTensor> col_offsets_ih, Rcpp::XPtr<XPtrTorchTensor> col_offsets_hh, Rcpp::XPtr<XPtrTorchScalar> scale_ih, Rcpp::XPtr<XPtrTorchScalar> scale_hh, Rcpp::XPtr<XPtrTorchScalar> zero_point_ih, Rcpp::XPtr<XPtrTorchScalar> zero_point_hh) {
  auto r_out = lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input->get(), hx->get(), w_ih->get(), w_hh->get(), b_ih->get(), b_hh->get(), packed_ih->get(), packed_hh->get(), col_offsets_ih->get(), col_offsets_hh->get(), scale_ih->get(), scale_hh->get(), zero_point_ih->get(), zero_point_hh->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pack_padded_sequence_input_Tensor_lengths_Tensor_batch_first_bool (Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> lengths, bool batch_first) {
  auto r_out = lantern__pack_padded_sequence_tensor_tensor_bool(input->get(), lengths->get(), reinterpret_cast<void*>(&batch_first));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pack_padded_sequence_backward_grad_Tensor_input_size_IntArrayRef_batch_sizes_Tensor_batch_first_bool (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> input_size, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, bool batch_first) {
  auto r_out = lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), batch_sizes->get(), reinterpret_cast<void*>(&batch_first));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pad_packed_sequence_data_Tensor_batch_sizes_Tensor_batch_first_bool_padding_value_Scalar_total_length_int64_t (Rcpp::XPtr<XPtrTorchTensor> data, Rcpp::XPtr<XPtrTorchTensor> batch_sizes, bool batch_first, Rcpp::XPtr<XPtrTorchScalar> padding_value, nullable<int64_t> total_length) {
  auto r_out = lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt(data->get(), batch_sizes->get(), reinterpret_cast<void*>(&batch_first), padding_value->get(), total_length.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_masked_fill_tensor_tensor_scalar(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_masked_fill_tensor_tensor_tensor(self->get(), mask->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_masked_scatter_tensor_tensor_tensor(self->get(), mask->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_index_add_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern_index_add_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_index_fill_tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_index_fill_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_index_fill_tensor_dimname_tensor_scalar(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> value) {
  auto r_out = lantern_index_fill_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_scatter_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_scatter_tensor_intt_tensor_scalar(self->get(), dim.get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_scatter_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_scatter_tensor_dimname_tensor_scalar(self->get(), dim->get(), index->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_scatter_add_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> src) {
  auto r_out = lantern_scatter_add_tensor_dimname_tensor_tensor(self->get(), dim->get(), index->get(), src->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_and_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_and_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_and_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_and_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___and___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern___and___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___and___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern___and___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_or_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_or_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_or_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_or_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___or___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern___or___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___or___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern___or___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_xor_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_xor_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_bitwise_xor_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_bitwise_xor_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___xor___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern___xor___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___xor___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern___xor___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___lshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern___lshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___lshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern___lshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___rshift___self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern___rshift___tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___rshift___self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern___rshift___tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out->get(), self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> batch1, Rcpp::XPtr<XPtrTorchTensor> batch2, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern_addbmm_tensor_tensor_tensor_scalar_scalar(self->get(), batch1->get(), batch2->get(), beta->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_diag_out_tensor_tensor_intt(out->get(), self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_diag_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_backward_grad_Tensor_input_sizes_IntArrayRef_diagonal_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> input_sizes, nullable<int64_t> diagonal) {
  auto r_out = lantern_diag_backward_tensor_intarrayref_intt(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_sizes.data(), input_sizes.size())).get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cross_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, nullable<int64_t> dim) {
  auto r_out = lantern_cross_out_tensor_tensor_tensor_intt(out->get(), self->get(), other->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cross_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, nullable<int64_t> dim) {
  auto r_out = lantern_cross_tensor_tensor_intt(self->get(), other->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_triu_out_tensor_tensor_intt(out->get(), self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_triu_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_tril_out_tensor_tensor_intt(out->get(), self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> diagonal) {
  auto r_out = lantern_tril_tensor_intt(self->get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_indices_row_int64_t_col_int64_t (nullable<int64_t> row, nullable<int64_t> col, nullable<int64_t> offset, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_tril_indices_intt_intt_intt_tensoroptions(row.get(), col.get(), offset.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_indices_row_int64_t_col_int64_t (nullable<int64_t> row, nullable<int64_t> col, nullable<int64_t> offset, Rcpp::XPtr<XPtrTorchTensorOptions> options) {
  auto r_out = lantern_triu_indices_intt_intt_intt_tensoroptions(row.get(), col.get(), offset.get(), options->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trace_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_trace_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trace_backward_grad_Tensor_sizes_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> sizes) {
  auto r_out = lantern_trace_backward_tensor_intarrayref(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(sizes.data(), sizes.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_ne_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_ne_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_ne_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_ne_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_not_equal_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_not_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_not_equal_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_not_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_eq_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_eq_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_eq_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_eq_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_ge_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_ge_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_ge_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_ge_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_greater_equal_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_greater_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_greater_equal_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_greater_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_le_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_le_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_le_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_le_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_less_equal_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_less_equal_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_less_equal_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_less_equal_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_gt_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_gt_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_gt_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_gt_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_greater_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_greater_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_greater_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_greater_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_lt_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_lt_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_lt_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_lt_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_less_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_less_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_less_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_less_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_out_out_Tensor_self_Tensor_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_take_out_tensor_tensor_tensor(out->get(), self->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_self_Tensor_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_take_tensor_tensor(self->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_backward_grad_Tensor_input_Tensor_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_take_backward_tensor_tensor_tensor(grad->get(), input->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_index_select_out_tensor_tensor_intt_tensor(out->get(), self->get(), dim.get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_index_select_tensor_intt_tensor(self->get(), dim.get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_index_select_out_tensor_tensor_dimname_tensor(out->get(), self->get(), dim->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_index_select_tensor_dimname_tensor(self->get(), dim->get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_backward_grad_Tensor_self_sizes_IntArrayRef_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, std::vector<int64_t> self_sizes, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index) {
  auto r_out = lantern_index_select_backward_tensor_intarrayref_intt_tensor(grad->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(self_sizes.data(), self_sizes.size())).get(), dim.get(), index->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_out_out_Tensor_self_Tensor_mask_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask) {
  auto r_out = lantern_masked_select_out_tensor_tensor_tensor(out->get(), self->get(), mask->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_self_Tensor_mask_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> mask) {
  auto r_out = lantern_masked_select_tensor_tensor(self->get(), mask->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_backward_grad_Tensor_input_Tensor_mask_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> mask) {
  auto r_out = lantern_masked_select_backward_tensor_tensor_tensor(grad->get(), input->get(), mask->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nonzero_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_nonzero_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nonzero_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_nonzero_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace_nonzero_numpy_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_nonzero_numpy_tensor(self->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_gather_out_tensor_tensor_intt_tensor_bool(out->get(), self->get(), dim.get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_gather_tensor_intt_tensor_bool(self->get(), dim.get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_backward_grad_Tensor_self_Tensor_dim_int64_t_index_Tensor_sparse_grad_bool (Rcpp::XPtr<XPtrTorchTensor> grad, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_gather_backward_tensor_tensor_intt_tensor_bool(grad->get(), self->get(), dim.get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_gather_out_tensor_tensor_dimname_tensor_bool(out->get(), self->get(), dim->get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, Rcpp::XPtr<XPtrTorchTensor> index, bool sparse_grad) {
  auto r_out = lantern_gather_tensor_dimname_tensor_bool(self->get(), dim->get(), index->get(), reinterpret_cast<void*>(&sparse_grad));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__gather_sparse_backward_self_Tensor_dim_int64_t_index_Tensor_grad_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> grad) {
  auto r_out = lantern__gather_sparse_backward_tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), grad->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcmul_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(out->get(), self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_addcmul_tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcdiv_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(out->get(), self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> tensor1, Rcpp::XPtr<XPtrTorchTensor> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern_addcdiv_tensor_tensor_tensor_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_out_X_Tensor_qr_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> X, Rcpp::XPtr<XPtrTorchTensor> qr, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_lstsq_out_tensor_tensor_tensor_tensor(X->get(), qr->get(), self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_lstsq_tensor_tensor(self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_out_X_Tensor_M_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> X, Rcpp::XPtr<XPtrTorchTensor> M, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool(X->get(), M->get(), self->get(), A->get(), reinterpret_cast<void*>(&upper), reinterpret_cast<void*>(&transpose), reinterpret_cast<void*>(&unitriangular));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = lantern_triangular_solve_tensor_tensor_bool_bool_bool(self->get(), A->get(), reinterpret_cast<void*>(&upper), reinterpret_cast<void*>(&transpose), reinterpret_cast<void*>(&unitriangular));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__triangular_solve_helper_self_Tensor_A_Tensor_upper_bool_transpose_bool_unitriangular_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = lantern__triangular_solve_helper_tensor_tensor_bool_bool_bool(self->get(), A->get(), reinterpret_cast<void*>(&upper), reinterpret_cast<void*>(&transpose), reinterpret_cast<void*>(&unitriangular));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_out_e_Tensor_V_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> e, Rcpp::XPtr<XPtrTorchTensor> V, Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors, bool upper) {
  auto r_out = lantern_symeig_out_tensor_tensor_tensor_bool_bool(e->get(), V->get(), self->get(), reinterpret_cast<void*>(&eigenvectors), reinterpret_cast<void*>(&upper));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors, bool upper) {
  auto r_out = lantern_symeig_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&eigenvectors), reinterpret_cast<void*>(&upper));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__symeig_helper_self_Tensor_eigenvectors_bool_upper_bool (Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors, bool upper) {
  auto r_out = lantern__symeig_helper_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&eigenvectors), reinterpret_cast<void*>(&upper));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_out_e_Tensor_v_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> e, Rcpp::XPtr<XPtrTorchTensor> v, Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors) {
  auto r_out = lantern_eig_out_tensor_tensor_tensor_bool(e->get(), v->get(), self->get(), reinterpret_cast<void*>(&eigenvectors));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool eigenvectors) {
  auto r_out = lantern_eig_tensor_bool(self->get(), reinterpret_cast<void*>(&eigenvectors));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_out_U_Tensor_S_Tensor_V_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> U, Rcpp::XPtr<XPtrTorchTensor> S, Rcpp::XPtr<XPtrTorchTensor> V, Rcpp::XPtr<XPtrTorchTensor> self, bool some, bool compute_uv) {
  auto r_out = lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool(U->get(), S->get(), V->get(), self->get(), reinterpret_cast<void*>(&some), reinterpret_cast<void*>(&compute_uv));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool some, bool compute_uv) {
  auto r_out = lantern_svd_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&some), reinterpret_cast<void*>(&compute_uv));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__svd_helper_self_Tensor_some_bool_compute_uv_bool (Rcpp::XPtr<XPtrTorchTensor> self, bool some, bool compute_uv) {
  auto r_out = lantern__svd_helper_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&some), reinterpret_cast<void*>(&compute_uv));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_cholesky_out_tensor_tensor_bool(out->get(), self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_cholesky_tensor_bool(self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cholesky_helper_self_Tensor_upper_bool (Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern__cholesky_helper_tensor_bool(self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_solve_out_out_Tensor_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, bool upper) {
  auto r_out = lantern_cholesky_solve_out_tensor_tensor_tensor_bool(out->get(), self->get(), input2->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_solve_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, bool upper) {
  auto r_out = lantern_cholesky_solve_tensor_tensor_bool(self->get(), input2->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cholesky_solve_helper_self_Tensor_A_Tensor_upper_bool (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A, bool upper) {
  auto r_out = lantern__cholesky_solve_helper_tensor_tensor_bool(self->get(), A->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_solve_tensor_tensor(self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_out_solution_Tensor_lu_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> solution, Rcpp::XPtr<XPtrTorchTensor> lu, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern_solve_out_tensor_tensor_tensor_tensor(solution->get(), lu->get(), self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__solve_helper_self_Tensor_A_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> A) {
  auto r_out = lantern__solve_helper_tensor_tensor(self->get(), A->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_inverse_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_cholesky_inverse_out_tensor_tensor_bool(out->get(), self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_inverse_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool upper) {
  auto r_out = lantern_cholesky_inverse_tensor_bool(self->get(), reinterpret_cast<void*>(&upper));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_out_Q_Tensor_R_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> Q, Rcpp::XPtr<XPtrTorchTensor> R, Rcpp::XPtr<XPtrTorchTensor> self, bool some) {
  auto r_out = lantern_qr_out_tensor_tensor_tensor_bool(Q->get(), R->get(), self->get(), reinterpret_cast<void*>(&some));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool some) {
  auto r_out = lantern_qr_tensor_bool(self->get(), reinterpret_cast<void*>(&some));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__qr_helper_self_Tensor_some_bool (Rcpp::XPtr<XPtrTorchTensor> self, bool some) {
  auto r_out = lantern__qr_helper_tensor_bool(self->get(), reinterpret_cast<void*>(&some));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_out_a_Tensor_tau_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> a, Rcpp::XPtr<XPtrTorchTensor> tau, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_geqrf_out_tensor_tensor_tensor(a->get(), tau->get(), self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_geqrf_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_orgqr_out_out_Tensor_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2) {
  auto r_out = lantern_orgqr_out_tensor_tensor_tensor(out->get(), self->get(), input2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_orgqr_self_Tensor_input2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2) {
  auto r_out = lantern_orgqr_tensor_tensor(self->get(), input2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ormqr_out_out_Tensor_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> input3, bool left, bool transpose) {
  auto r_out = lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool(out->get(), self->get(), input2->get(), input3->get(), reinterpret_cast<void*>(&left), reinterpret_cast<void*>(&transpose));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ormqr_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> input2, Rcpp::XPtr<XPtrTorchTensor> input3, bool left, bool transpose) {
  auto r_out = lantern_ormqr_tensor_tensor_tensor_bool_bool(self->get(), input2->get(), input3->get(), reinterpret_cast<void*>(&left), reinterpret_cast<void*>(&transpose));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__lu_with_info_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool pivot, bool check_errors) {
  auto r_out = lantern__lu_with_info_tensor_bool_bool(self->get(), reinterpret_cast<void*>(&pivot), reinterpret_cast<void*>(&check_errors));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lu_solve_out_out_Tensor_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> LU_data, Rcpp::XPtr<XPtrTorchTensor> LU_pivots) {
  auto r_out = lantern_lu_solve_out_tensor_tensor_tensor_tensor(out->get(), self->get(), LU_data->get(), LU_pivots->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> LU_data, Rcpp::XPtr<XPtrTorchTensor> LU_pivots) {
  auto r_out = lantern_lu_solve_tensor_tensor_tensor(self->get(), LU_data->get(), LU_pivots->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__lu_solve_helper_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> LU_data, Rcpp::XPtr<XPtrTorchTensor> LU_pivots) {
  auto r_out = lantern__lu_solve_helper_tensor_tensor_tensor(self->get(), LU_data->get(), LU_pivots->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multinomial_out_out_Tensor_self_Tensor_num_samples_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> num_samples, bool replacement, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_multinomial_out_tensor_tensor_intt_bool_generator(out->get(), self->get(), num_samples.get(), reinterpret_cast<void*>(&replacement), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multinomial_self_Tensor_num_samples_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> num_samples, bool replacement, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_multinomial_tensor_intt_bool_generator(self->get(), num_samples.get(), reinterpret_cast<void*>(&replacement), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__multinomial_alias_setup_probs_Tensor (Rcpp::XPtr<XPtrTorchTensor> probs) {
  auto r_out = lantern__multinomial_alias_setup_tensor(probs->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__multinomial_alias_draw_J_Tensor_q_Tensor_num_samples_int64_t (Rcpp::XPtr<XPtrTorchTensor> J, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> num_samples, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern__multinomial_alias_draw_tensor_tensor_intt_generator(J->get(), q->get(), num_samples.get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lgamma_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_lgamma_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lgamma_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_lgamma_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_digamma_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_digamma_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_digamma_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_digamma_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polygamma_out_out_Tensor_n_int64_t_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, nullable<int64_t> n, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_polygamma_out_tensor_intt_tensor(out->get(), n.get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfinv_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erfinv_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfinv_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_erfinv_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_i0_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_i0__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_i0_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sign_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sign_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sign_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_sign_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_signbit_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_signbit_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_signbit_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_signbit_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dist_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> p) {
  auto r_out = lantern_dist_tensor_tensor_scalar(self->get(), other->get(), p->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan2_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_atan2_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan2_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_atan2_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchScalar> weight) {
  auto r_out = lantern_lerp_out_tensor_tensor_tensor_scalar(out->get(), self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_lerp_out_tensor_tensor_tensor_tensor(out->get(), self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchScalar> weight) {
  auto r_out = lantern_lerp_tensor_tensor_scalar(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> end, Rcpp::XPtr<XPtrTorchTensor> weight) {
  auto r_out = lantern_lerp_tensor_tensor_tensor(self->get(), end->get(), weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_histc_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> bins, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_histc_out_tensor_tensor_intt_scalar_scalar(out->get(), self->get(), bins.get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_histc_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> bins, Rcpp::XPtr<XPtrTorchScalar> min, Rcpp::XPtr<XPtrTorchScalar> max) {
  auto r_out = lantern_histc_tensor_intt_scalar_scalar(self->get(), bins.get(), min->get(), max->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_fmod_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_fmod_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_fmod_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_fmod_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hypot_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_hypot_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hypot_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_hypot_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nextafter_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_nextafter_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nextafter_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_nextafter_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_remainder_out_tensor_tensor_scalar(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_self_Tensor_other_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> other) {
  auto r_out = lantern_remainder_tensor_scalar(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_remainder_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_remainder_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_min_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_max_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_maximum_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_maximum_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_maximum_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_maximum_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_max_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_max_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_minimum_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_minimum_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_minimum_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_minimum_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_min_out_tensor_tensor_tensor(out->get(), self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_min_tensor_tensor(self->get(), other->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_median_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_median_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_quantile_out_tensor_tensor_double_intt_bool(out->get(), self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_quantile_tensor_double_intt_bool(self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_quantile_out_tensor_tensor_tensor_intt_bool(out->get(), self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_quantile_tensor_tensor_intt_bool(self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_double_intt_bool(out->get(), self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_double (Rcpp::XPtr<XPtrTorchTensor> self, double q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nanquantile_tensor_double_intt_bool(self->get(), XPtrTorch(lantern_double(q)).get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_tensor_intt_bool(out->get(), self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> q, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern_nanquantile_tensor_tensor_intt_bool(self->get(), q->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_intt_bool(values->get(), indices->get(), self->get(), dim.get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool descending) {
  auto r_out = lantern_sort_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_dimname_bool(values->get(), indices->get(), self->get(), dim->get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool descending) {
  auto r_out = lantern_sort_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&descending));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argsort_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool descending) {
  auto r_out = lantern_argsort_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&descending));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argsort_self_Tensor_dim_Dimname (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchDimname> dim, bool descending) {
  auto r_out = lantern_argsort_tensor_dimname_bool(self->get(), dim->get(), reinterpret_cast<void*>(&descending));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool largest, bool sorted) {
  auto r_out = lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool(values->get(), indices->get(), self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&largest), reinterpret_cast<void*>(&sorted));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_self_Tensor_k_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> k, nullable<int64_t> dim, bool largest, bool sorted) {
  auto r_out = lantern_topk_tensor_intt_intt_bool_bool(self->get(), k.get(), dim.get(), reinterpret_cast<void*>(&largest), reinterpret_cast<void*>(&sorted));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_all_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_any_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_renorm_out_out_Tensor_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchScalar> maxnorm) {
  auto r_out = lantern_renorm_out_tensor_tensor_scalar_intt_scalar(out->get(), self->get(), p->get(), dim.get(), maxnorm->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> p, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchScalar> maxnorm) {
  auto r_out = lantern_renorm_tensor_scalar_intt_scalar(self->get(), p->get(), dim.get(), maxnorm->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_unfold_backward_grad_in_Tensor_input_sizes_IntArrayRef_dim_int64_t_size_int64_t_step_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_in, std::vector<int64_t> input_sizes, nullable<int64_t> dim, nullable<int64_t> size, nullable<int64_t> step) {
  auto r_out = lantern_unfold_backward_tensor_intarrayref_intt_intt_intt(grad_in->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_sizes.data(), input_sizes.size())).get(), dim.get(), size.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_equal_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other) {
  auto r_out = lantern_equal_tensor_tensor(self->get(), other->get());
return reinterpret_and_clean<bool, lantern_bool_delete>(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_pow_out_tensor_tensor_tensor(out->get(), self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Tensor_exponent_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_pow_tensor_tensor(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Scalar_exponent_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchScalar> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_pow_out_tensor_scalar_tensor(out->get(), self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Scalar_exponent_Tensor (Rcpp::XPtr<XPtrTorchScalar> self, Rcpp::XPtr<XPtrTorchTensor> exponent) {
  auto r_out = lantern_pow_scalar_tensor(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Scalar (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> exponent) {
  auto r_out = lantern_pow_out_tensor_tensor_scalar(out->get(), self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Tensor_exponent_Scalar (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> exponent) {
  auto r_out = lantern_pow_tensor_scalar(self->get(), exponent->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> mean, double std, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_normal_out_tensor_tensor_double_generator(out->get(), mean->get(), XPtrTorch(lantern_double(std)).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, double mean, Rcpp::XPtr<XPtrTorchTensor> std, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_normal_out_tensor_double_tensor_generator(out->get(), XPtrTorch(lantern_double(mean)).get(), std->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor_std_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> mean, Rcpp::XPtr<XPtrTorchTensor> std, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_normal_out_tensor_tensor_tensor_generator(out->get(), mean->get(), std->get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_double_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, double mean, double std, std::vector<int64_t> size, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_normal_out_tensor_double_double_intarrayref_generator(out->get(), XPtrTorch(lantern_double(mean)).get(), XPtrTorch(lantern_double(std)).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alias_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_alias_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, Rcpp::XPtr<XPtrTorchTensor> index, Rcpp::XPtr<XPtrTorchTensor> source) {
  auto r_out = lantern__index_copy__tensor_intt_tensor_tensor(self->get(), dim.get(), index->get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__cumsum_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cumsum_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__cumsum_out_tensor_tensor_intt(out->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__cumprod_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cumprod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern__cumprod_out_tensor_tensor_intt(out->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__var_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern__var_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__std_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, bool unbiased) {
  auto r_out = lantern__std_tensor_bool(self->get(), reinterpret_cast<void*>(&unbiased));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__amp_non_finite_check_and_unscale__self_Tensor_found_inf_Tensor_inv_scale_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> found_inf, Rcpp::XPtr<XPtrTorchTensor> inv_scale) {
  lantern__amp_non_finite_check_and_unscale__tensor_tensor_tensor(self->get(), found_inf->get(), inv_scale->get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__amp_update_scale_growth_tracker_Tensor_current_scale_Tensor_found_inf_Tensor_scale_growth_factor_double_scale_backoff_factor_double_growth_interval_int64_t (Rcpp::XPtr<XPtrTorchTensor> growth_tracker, Rcpp::XPtr<XPtrTorchTensor> current_scale, Rcpp::XPtr<XPtrTorchTensor> found_inf, double scale_growth_factor, double scale_backoff_factor, nullable<int64_t> growth_interval) {
  auto r_out = lantern__amp_update_scale_tensor_tensor_tensor_double_double_intt(growth_tracker->get(), current_scale->get(), found_inf->get(), XPtrTorch(lantern_double(scale_growth_factor)).get(), XPtrTorch(lantern_double(scale_backoff_factor)).get(), growth_interval.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cat_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern__cat_tensorlist_intt(tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cat_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensorList> tensors, nullable<int64_t> dim) {
  auto r_out = lantern__cat_out_tensor_tensorlist_intt(out->get(), tensors->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_add_tensors_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  auto r_out = lantern__foreach_add_tensorlist_scalar(tensors->get(), scalar->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add__self_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  lantern__foreach_add__tensorlist_scalar(self->get(), scalar->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_sub_tensors_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  auto r_out = lantern__foreach_sub_tensorlist_scalar(tensors->get(), scalar->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub__self_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  lantern__foreach_sub__tensorlist_scalar(self->get(), scalar->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_mul_tensors_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  auto r_out = lantern__foreach_mul_tensorlist_scalar(tensors->get(), scalar->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul__self_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  lantern__foreach_mul__tensorlist_scalar(self->get(), scalar->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_div_tensors_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> tensors, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  auto r_out = lantern__foreach_div_tensorlist_scalar(tensors->get(), scalar->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div__self_TensorList_scalar_Scalar (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchScalar> scalar) {
  lantern__foreach_div__tensorlist_scalar(self->get(), scalar->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_add_tensors1_TensorList_tensors2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors1, Rcpp::XPtr<XPtrTorchTensorList> tensors2, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__foreach_add_tensorlist_tensorlist_scalar(tensors1->get(), tensors2->get(), alpha->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add__self_TensorList_other_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  lantern__foreach_add__tensorlist_tensorlist_scalar(self->get(), other->get(), alpha->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_sub_tensors1_TensorList_tensors2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors1, Rcpp::XPtr<XPtrTorchTensorList> tensors2, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__foreach_sub_tensorlist_tensorlist_scalar(tensors1->get(), tensors2->get(), alpha->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub__self_TensorList_other_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  lantern__foreach_sub__tensorlist_tensorlist_scalar(self->get(), other->get(), alpha->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_mul_tensors1_TensorList_tensors2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors1, Rcpp::XPtr<XPtrTorchTensorList> tensors2) {
  auto r_out = lantern__foreach_mul_tensorlist_tensorlist(tensors1->get(), tensors2->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul__self_TensorList_other_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> other) {
  lantern__foreach_mul__tensorlist_tensorlist(self->get(), other->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_div_tensors1_TensorList_tensors2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors1, Rcpp::XPtr<XPtrTorchTensorList> tensors2) {
  auto r_out = lantern__foreach_div_tensorlist_tensorlist(tensors1->get(), tensors2->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div__self_TensorList_other_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> other) {
  lantern__foreach_div__tensorlist_tensorlist(self->get(), other->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_add_scalar_list_tensors_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> tensors, std::vector<double> scalars) {
  auto r_out = lantern__foreach_add_scalar_list_tensorlist_arrayrefdouble(tensors->get(), lantern_vector_double(scalars.data(), scalars.size()));
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add_scalar_list__self_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> self, std::vector<double> scalars) {
  lantern__foreach_add_scalar_list__tensorlist_arrayrefdouble(self->get(), lantern_vector_double(scalars.data(), scalars.size()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_sub_scalar_list_tensors_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> tensors, std::vector<double> scalars) {
  auto r_out = lantern__foreach_sub_scalar_list_tensorlist_arrayrefdouble(tensors->get(), lantern_vector_double(scalars.data(), scalars.size()));
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub_scalar_list__self_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> self, std::vector<double> scalars) {
  lantern__foreach_sub_scalar_list__tensorlist_arrayrefdouble(self->get(), lantern_vector_double(scalars.data(), scalars.size()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_div_scalar_list_tensors_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> tensors, std::vector<double> scalars) {
  auto r_out = lantern__foreach_div_scalar_list_tensorlist_arrayrefdouble(tensors->get(), lantern_vector_double(scalars.data(), scalars.size()));
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div_scalar_list__self_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> self, std::vector<double> scalars) {
  lantern__foreach_div_scalar_list__tensorlist_arrayrefdouble(self->get(), lantern_vector_double(scalars.data(), scalars.size()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_mul_scalar_list_tensors_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> tensors, std::vector<double> scalars) {
  auto r_out = lantern__foreach_mul_scalar_list_tensorlist_arrayrefdouble(tensors->get(), lantern_vector_double(scalars.data(), scalars.size()));
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul_scalar_list__self_TensorList_scalars_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensorList> self, std::vector<double> scalars) {
  lantern__foreach_mul_scalar_list__tensorlist_arrayrefdouble(self->get(), lantern_vector_double(scalars.data(), scalars.size()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_exp_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern__foreach_exp_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_exp__self_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self) {
  lantern__foreach_exp__tensorlist(self->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_sqrt_tensors_TensorList (Rcpp::XPtr<XPtrTorchTensorList> tensors) {
  auto r_out = lantern__foreach_sqrt_tensorlist(tensors->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sqrt__self_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self) {
  lantern__foreach_sqrt__tensorlist(self->get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcdiv__self_TensorList_tensor1_TensorList_tensor2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> tensor1, Rcpp::XPtr<XPtrTorchTensorList> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  lantern__foreach_addcdiv__tensorlist_tensorlist_tensorlist_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcmul__self_TensorList_tensor1_TensorList_tensor2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> self, Rcpp::XPtr<XPtrTorchTensorList> tensor1, Rcpp::XPtr<XPtrTorchTensorList> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  lantern__foreach_addcmul__tensorlist_tensorlist_tensorlist_scalar(self->get(), tensor1->get(), tensor2->get(), value->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_addcdiv_input_TensorList_tensor1_TensorList_tensor2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> input, Rcpp::XPtr<XPtrTorchTensorList> tensor1, Rcpp::XPtr<XPtrTorchTensorList> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern__foreach_addcdiv_tensorlist_tensorlist_tensorlist_scalar(input->get(), tensor1->get(), tensor2->get(), value->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_namespace__foreach_addcmul_input_TensorList_tensor1_TensorList_tensor2_TensorList (Rcpp::XPtr<XPtrTorchTensorList> input, Rcpp::XPtr<XPtrTorchTensorList> tensor1, Rcpp::XPtr<XPtrTorchTensorList> tensor2, Rcpp::XPtr<XPtrTorchScalar> value) {
  auto r_out = lantern__foreach_addcmul_tensorlist_tensorlist_tensorlist_scalar(input->get(), tensor1->get(), tensor2->get(), value->get());
return make_xptr<XPtrTorchTensorList>(r_out, "TensorList");
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__mode_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern__mode_tensor_intt_bool(self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__mode_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> values, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim, bool keepdim) {
  auto r_out = lantern__mode_out_tensor_tensor_tensor_intt_bool(values->get(), indices->get(), self->get(), dim.get(), reinterpret_cast<void*>(&keepdim));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_self_Tensor_boundaries_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> boundaries, bool out_int32, bool right) {
  auto r_out = lantern_bucketize_tensor_tensor_bool_bool(self->get(), boundaries->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_out_out_Tensor_self_Tensor_boundaries_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> boundaries, bool out_int32, bool right) {
  auto r_out = lantern_bucketize_out_tensor_tensor_tensor_bool_bool(out->get(), self->get(), boundaries->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_self_Scalar_boundaries_Tensor (Rcpp::XPtr<XPtrTorchScalar> self, Rcpp::XPtr<XPtrTorchTensor> boundaries, bool out_int32, bool right) {
  auto r_out = lantern_bucketize_scalar_tensor_bool_bool(self->get(), boundaries->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_sorted_sequence_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> sorted_sequence, Rcpp::XPtr<XPtrTorchTensor> self, bool out_int32, bool right) {
  auto r_out = lantern_searchsorted_tensor_tensor_bool_bool(sorted_sequence->get(), self->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_out_out_Tensor_sorted_sequence_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> sorted_sequence, Rcpp::XPtr<XPtrTorchTensor> self, bool out_int32, bool right) {
  auto r_out = lantern_searchsorted_out_tensor_tensor_tensor_bool_bool(out->get(), sorted_sequence->get(), self->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_sorted_sequence_Tensor_self_Scalar (Rcpp::XPtr<XPtrTorchTensor> sorted_sequence, Rcpp::XPtr<XPtrTorchScalar> self, bool out_int32, bool right) {
  auto r_out = lantern_searchsorted_tensor_scalar_bool_bool(sorted_sequence->get(), self->get(), reinterpret_cast<void*>(&out_int32), reinterpret_cast<void*>(&right));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_mse_loss_out_tensor_tensor_tensor_intt(out->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_mse_loss_tensor_tensor_intt(self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input->get(), grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_mse_loss_backward_tensor_tensor_tensor_intt(grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_l1_loss_out_tensor_tensor_tensor_intt(out->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_l1_loss_tensor_tensor_intt(self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input->get(), grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_l1_loss_backward_tensor_tensor_tensor_intt(grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorchScalar> margin, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt(out->get(), self->get(), target->get(), p->get(), margin->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorchScalar> margin, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt(self->get(), target->get(), p->get(), margin->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorchScalar> margin, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt(grad_input->get(), grad_output->get(), self->get(), target->get(), p->get(), margin->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchScalar> p, Rcpp::XPtr<XPtrTorchScalar> margin, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction) {
  auto r_out = lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt(grad_output->get(), self->get(), target->get(), p->get(), margin->get(), weight->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(out->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_multilabel_margin_loss_tensor_tensor_intt(self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_out_output_Tensor_is_target_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> is_target, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt(output->get(), is_target->get(), self->get(), target->get(), reduction.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_multilabel_margin_loss_forward_tensor_tensor_intt(self->get(), target->get(), reduction.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, Rcpp::XPtr<XPtrTorchTensor> is_target) {
  auto r_out = lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor(grad_input->get(), grad_output->get(), self->get(), target->get(), reduction.get(), is_target->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, Rcpp::XPtr<XPtrTorchTensor> is_target) {
  auto r_out = lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor(grad_output->get(), self->get(), target->get(), reduction.get(), is_target->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(out->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss_tensor_tensor_tensor_intt_intt(self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> total_weight, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(output->get(), total_weight->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt(self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index, Rcpp::XPtr<XPtrTorchTensor> total_weight) {
  auto r_out = lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_input->get(), grad_output->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get(), total_weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index, Rcpp::XPtr<XPtrTorchTensor> total_weight) {
  auto r_out = lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_output->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get(), total_weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(out->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> total_weight, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(output->get(), total_weight->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index) {
  auto r_out = lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt(self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index, Rcpp::XPtr<XPtrTorchTensor> total_weight) {
  auto r_out = lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_input->get(), grad_output->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get(), total_weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, Rcpp::XPtr<XPtrTorchTensor> weight, nullable<int64_t> reduction, nullable<int64_t> ignore_index, Rcpp::XPtr<XPtrTorchTensor> total_weight) {
  auto r_out = lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_output->get(), self->get(), target->get(), weight->get(), reduction.get(), ignore_index.get(), total_weight->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, double beta) {
  auto r_out = lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt_double(out->get(), self->get(), target->get(), reduction.get(), XPtrTorch(lantern_double(beta)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, double beta) {
  auto r_out = lantern_smooth_l1_loss_tensor_tensor_intt_double(self->get(), target->get(), reduction.get(), XPtrTorch(lantern_double(beta)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_beta_double (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, double beta) {
  auto r_out = lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(grad_input->get(), grad_output->get(), self->get(), target->get(), reduction.get(), XPtrTorch(lantern_double(beta)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_beta_double (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction, double beta) {
  auto r_out = lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt_double(grad_output->get(), self->get(), target->get(), reduction.get(), XPtrTorch(lantern_double(beta)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_soft_margin_loss_out_tensor_tensor_tensor_intt(out->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_soft_margin_loss_tensor_tensor_intt(self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input->get(), grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> target, nullable<int64_t> reduction) {
  auto r_out = lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt(grad_output->get(), self->get(), target->get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> alpha, Rcpp::XPtr<XPtrTorchScalar> scale, Rcpp::XPtr<XPtrTorchScalar> input_scale) {
  auto r_out = lantern_elu_out_tensor_tensor_scalar_scalar_scalar(out->get(), self->get(), alpha->get(), scale->get(), input_scale->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> alpha, Rcpp::XPtr<XPtrTorchScalar> scale, Rcpp::XPtr<XPtrTorchScalar> input_scale) {
  auto r_out = lantern_elu_tensor_scalar_scalar_scalar(self->get(), alpha->get(), scale->get(), input_scale->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_backward_out_grad_input_Tensor_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchScalar> alpha, Rcpp::XPtr<XPtrTorchScalar> scale, Rcpp::XPtr<XPtrTorchScalar> input_scale, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_tensor(grad_input->get(), grad_output->get(), alpha->get(), scale->get(), input_scale->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_backward_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchScalar> alpha, Rcpp::XPtr<XPtrTorchScalar> scale, Rcpp::XPtr<XPtrTorchScalar> input_scale, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_elu_backward_tensor_scalar_scalar_scalar_tensor(grad_output->get(), alpha->get(), scale->get(), input_scale->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> alpha, Rcpp::XPtr<XPtrTorchScalar> scale, Rcpp::XPtr<XPtrTorchScalar> input_scale) {
  auto r_out = lantern_elu__tensor_scalar_scalar_scalar(self->get(), alpha->get(), scale->get(), input_scale->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_glu_out_tensor_tensor_intt(out->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_glu_tensor_intt(self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_glu_backward_out_tensor_tensor_tensor_intt(grad_input->get(), grad_output->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_backward_grad_output_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> dim) {
  auto r_out = lantern_glu_backward_tensor_tensor_intt(grad_output->get(), self->get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardsigmoid_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardsigmoid_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardsigmoid__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardsigmoid_backward_tensor_tensor(grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min_val, Rcpp::XPtr<XPtrTorchScalar> max_val) {
  auto r_out = lantern_hardtanh_out_tensor_tensor_scalar_scalar(out->get(), self->get(), min_val->get(), max_val->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min_val, Rcpp::XPtr<XPtrTorchScalar> max_val) {
  auto r_out = lantern_hardtanh_tensor_scalar_scalar(self->get(), min_val->get(), max_val->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min_val, Rcpp::XPtr<XPtrTorchScalar> max_val) {
  auto r_out = lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar(grad_input->get(), grad_output->get(), self->get(), min_val->get(), max_val->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_backward_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min_val, Rcpp::XPtr<XPtrTorchScalar> max_val) {
  auto r_out = lantern_hardtanh_backward_tensor_tensor_scalar_scalar(grad_output->get(), self->get(), min_val->get(), max_val->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> min_val, Rcpp::XPtr<XPtrTorchScalar> max_val) {
  auto r_out = lantern_hardtanh__tensor_scalar_scalar(self->get(), min_val->get(), max_val->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardswish_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardswish_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardswish__tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_hardswish_backward_tensor_tensor(grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> negative_slope) {
  auto r_out = lantern_leaky_relu_out_tensor_tensor_scalar(out->get(), self->get(), negative_slope->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> negative_slope) {
  auto r_out = lantern_leaky_relu_tensor_scalar(self->get(), negative_slope->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_backward_grad_output_Tensor_self_Tensor_negative_slope_Scalar_self_is_result_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> negative_slope, bool self_is_result) {
  auto r_out = lantern_leaky_relu_backward_tensor_tensor_scalar_bool(grad_output->get(), self->get(), negative_slope->get(), reinterpret_cast<void*>(&self_is_result));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu__self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> negative_slope) {
  auto r_out = lantern_leaky_relu__tensor_scalar(self->get(), negative_slope->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_sigmoid_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_sigmoid_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_out_output_Tensor_buffer_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> buffer, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_sigmoid_forward_out_tensor_tensor_tensor(output->get(), buffer->get(), self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_log_sigmoid_forward_tensor(self->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_buffer_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> buffer) {
  auto r_out = lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor(grad_input->get(), grad_output->get(), self->get(), buffer->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_backward_grad_output_Tensor_self_Tensor_buffer_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> buffer) {
  auto r_out = lantern_log_sigmoid_backward_tensor_tensor_tensor(grad_output->get(), self->get(), buffer->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_out_out_Tensor_self_Tensor_noise_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> noise, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator(out->get(), self->get(), noise->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_self_Tensor_noise_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> noise, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator(self->get(), noise->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_backward_grad_output_Tensor_self_Tensor_noise_Tensor_lower_Scalar_upper_Scalar_training_bool_self_is_result_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> noise, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, bool self_is_result) {
  auto r_out = lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool_bool(grad_output->get(), self->get(), noise->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), reinterpret_cast<void*>(&self_is_result));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise__self_Tensor_noise_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> noise, Rcpp::XPtr<XPtrTorchScalar> lower, Rcpp::XPtr<XPtrTorchScalar> upper, bool training, Rcpp::XPtr<XPtrTorch> generator) {
  auto r_out = lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator(self->get(), noise->get(), lower->get(), upper->get(), reinterpret_cast<void*>(&training), generator->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> threshold) {
  auto r_out = lantern_softplus_out_tensor_tensor_scalar_scalar(out->get(), self->get(), beta->get(), threshold->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> threshold) {
  auto r_out = lantern_softplus_tensor_scalar_scalar(self->get(), beta->get(), threshold->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> threshold, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor(grad_input->get(), grad_output->get(), self->get(), beta->get(), threshold->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_backward_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> beta, Rcpp::XPtr<XPtrTorchScalar> threshold, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor(grad_output->get(), self->get(), beta->get(), threshold->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_softshrink_out_tensor_tensor_scalar(out->get(), self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_softshrink_tensor_scalar(self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_softshrink_backward_out_tensor_tensor_tensor_scalar(grad_input->get(), grad_output->get(), self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_backward_grad_output_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> lambd) {
  auto r_out = lantern_softshrink_backward_tensor_tensor_scalar(grad_output->get(), self->get(), lambd->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_avg_pool2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern__adaptive_avg_pool2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool2d_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern__adaptive_avg_pool2d_backward_tensor_tensor(grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_avg_pool3d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor(grad_input->get(), grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_adaptive_avg_pool3d_backward_tensor_tensor(grad_output->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref(out->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_max_pool2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor(grad_input->get(), grad_output->get(), self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor(grad_output->get(), self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref(out->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size) {
  auto r_out = lantern_adaptive_max_pool3d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor(grad_input->get(), grad_output->get(), self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor(grad_output->get(), self->get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, nullable<int64_t> divisor_override) {
  auto r_out = lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), reinterpret_cast<void*>(&ceil_mode), reinterpret_cast<void*>(&count_include_pad), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> random_samples) {
  auto r_out = lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(output->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), random_samples->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> random_samples) {
  auto r_out = lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), random_samples->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> random_samples) {
  auto r_out = lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(output->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), random_samples->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> random_samples) {
  auto r_out = lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), random_samples->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(out->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> indices, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(out->get(), indices->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<XPtrTorchTensor> indices) {
  auto r_out = lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&ceil_mode), indices->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size) {
  auto r_out = lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref(out->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size) {
  auto r_out = lantern_max_unpool2d_tensor_tensor_intarrayref(self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size) {
  auto r_out = lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size) {
  auto r_out = lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref(grad_output->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(out->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref(self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(grad_input->get(), grad_output->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(grad_output->get(), self->get(), indices->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad1d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad1d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad1d_backward_tensor_tensor_intarrayref(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad2d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_reflection_pad2d_backward_tensor_tensor_intarrayref(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad1d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad1d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad1d_backward_tensor_tensor_intarrayref(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad2d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad2d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad2d_backward_tensor_tensor_intarrayref(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad3d_out_tensor_tensor_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad3d_tensor_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref(grad_input->get(), grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> padding) {
  auto r_out = lantern_replication_pad3d_backward_tensor_tensor_intarrayref(grad_output->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_linear1d_tensor_intarrayref_bool_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_bilinear2d_tensor_intarrayref_bool_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_trilinear3d_tensor_intarrayref_bool_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_bicubic2d_tensor_intarrayref_bool_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, bool align_corners, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest1d_tensor_intarrayref_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest2d_tensor_intarrayref_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> input, nullableVector<std::vector<int64_t>> output_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest3d_tensor_intarrayref_arrayrefdouble(input->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> grad_output, nullableVector<std::vector<int64_t>> output_size, std::vector<int64_t> input_size, nullableVector<std::vector<double>> scale_factors) {
  auto r_out = lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output->get(), lantern_optional_vector_int64_t(output_size.x.data(), output_size.x.size(), output_size.is_null), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), lantern_optional_vector_double(scale_factors.x.data(), scale_factors.x.size(), scale_factors.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales) {
  auto r_out = lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales) {
  auto r_out = lantern_upsample_linear1d_tensor_intarrayref_bool_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales) {
  auto r_out = lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales) {
  auto r_out = lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool_double_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bilinear2d_tensor_intarrayref_bool_double_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_double_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool_double_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bicubic2d_tensor_intarrayref_bool_double_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_double_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool_double_double_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, bool align_corners, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_trilinear3d_tensor_intarrayref_bool_double_double_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_double_double_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), reinterpret_cast<void*>(&align_corners), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales) {
  auto r_out = lantern_upsample_nearest1d_out_tensor_tensor_intarrayref_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales) {
  auto r_out = lantern_upsample_nearest1d_tensor_intarrayref_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales) {
  auto r_out = lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales) {
  auto r_out = lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales.x, scales.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest2d_out_tensor_tensor_intarrayref_double_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest2d_tensor_intarrayref_double_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_double_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest3d_out_tensor_tensor_intarrayref_double_double_double(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest3d_tensor_intarrayref_double_double_double(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double_double(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, nullable<double> scales_d, nullable<double> scales_h, nullable<double> scales_w) {
  auto r_out = lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_double_double_double(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorch(lantern_optional_double(scales_d.x, scales_d.is_null)).get(), XPtrTorch(lantern_optional_double(scales_h.x, scales_h.is_null)).get(), XPtrTorch(lantern_optional_double(scales_w.x, scales_w.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_sigmoid_backward_out_tensor_tensor_tensor(grad_input->get(), grad_output->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_backward_grad_output_Tensor_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_sigmoid_backward_tensor_tensor(grad_output->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_logit_backward_out_tensor_tensor_tensor_double(grad_input->get(), grad_output->get(), self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, nullable<double> eps) {
  auto r_out = lantern_logit_backward_tensor_tensor_double(grad_output->get(), self->get(), XPtrTorch(lantern_optional_double(eps.x, eps.is_null)).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_tanh_backward_out_tensor_tensor_tensor(grad_input->get(), grad_output->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_backward_grad_output_Tensor_output_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> output) {
  auto r_out = lantern_tanh_backward_tensor_tensor(grad_output->get(), output->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_weight, Rcpp::XPtr<XPtrTorchTensor> grad_bias, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<XPtrTorchTensor> columns, Rcpp::XPtr<XPtrTorchTensor> ones) {
  auto r_out = lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input->get(), grad_weight->get(), grad_bias->get(), grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), columns->get(), ones->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<XPtrTorchTensor> columns, Rcpp::XPtr<XPtrTorchTensor> ones, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), columns->get(), ones->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_weight, Rcpp::XPtr<XPtrTorchTensor> grad_bias, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input) {
  auto r_out = lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input->get(), grad_weight->get(), grad_bias->get(), grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), finput->get(), fgrad_input->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_padding.data(), output_padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), finput->get(), fgrad_input->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_forward_out_output_Tensor_finput_Tensor_fgrad_input_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(output->get(), finput->get(), fgrad_input->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_weight, Rcpp::XPtr<XPtrTorchTensor> grad_bias, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input) {
  auto r_out = lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input->get(), grad_weight->get(), grad_bias->get(), grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), finput->get(), fgrad_input->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), finput->get(), fgrad_input->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv_depthwise2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv_depthwise2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv_depthwise2d_forward_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv_depthwise2d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv_depthwise2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_weight, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_input->get(), grad_weight->get(), grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv_depthwise2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool2 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_slow_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(out->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_slow_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_out_output_Tensor_finput_Tensor_fgrad_input_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> output, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_slow_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(output->get(), finput->get(), fgrad_input->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = lantern_slow_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_weight, Rcpp::XPtr<XPtrTorchTensor> grad_bias, Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input) {
  auto r_out = lantern_slow_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input->get(), grad_weight->get(), grad_bias->get(), grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), finput->get(), fgrad_input->get());
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<XPtrTorchTensor> finput, Rcpp::XPtr<XPtrTorchTensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), finput->get(), fgrad_input->get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_dilated2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_dilated3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<XPtrTorchTensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), bias->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<XPtrTorchTensor> grad_output, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output->get(), self->get(), weight->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), reinterpret_cast<void*>(&output_mask));
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(r_out, 0)),XPtrTorchTensor(lantern_vector_get(r_out, 1)),XPtrTorchTensor(lantern_vector_get(r_out, 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_out_out_Tensor_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> output_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(output_size.data(), output_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_backward_out_grad_input_Tensor_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_backward_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out->get(), self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_backward_out_grad_input_Tensor_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_input, Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(grad_input->get(), grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_backward_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> grad_output, std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(grad_output->get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(input_size.data(), input_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(kernel_size.data(), kernel_size.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(dilation.data(), dilation.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(padding.data(), padding.size())).get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(stride.data(), stride.size())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isfinite_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isfinite_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isinf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isinf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isposinf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isposinf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isposinf_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isposinf_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isneginf_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isneginf_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isneginf_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_isneginf_out_tensor_tensor(out->get(), self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_batch_dim_self_Tensor_batch_dim_int64_t_level_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> batch_dim, nullable<int64_t> level) {
  auto r_out = lantern__add_batch_dim_tensor_intt_intt(self->get(), batch_dim.get(), level.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__remove_batch_dim_self_Tensor_level_int64_t_batch_size_int64_t_out_dim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> level, nullable<int64_t> batch_size, nullable<int64_t> out_dim) {
  auto r_out = lantern__remove_batch_dim_tensor_intt_intt_intt(self->get(), level.get(), batch_size.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_fft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_ifft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_rfft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_irfft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_hfft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_hfft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ihfft_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> n, nullable<int64_t> dim, std::string norm) {
  auto r_out = lantern_fft_ihfft_tensor_intt_intt_stdstring(self->get(), n.get(), dim.get(), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullableVector<std::vector<int64_t>> s, nullableVector<std::vector<int64_t>> dim, std::string norm) {
  auto r_out = lantern_fft_fftn_tensor_intarrayref_intarrayref_stdstring(self->get(), lantern_optional_vector_int64_t(s.x.data(), s.x.size(), s.is_null), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifftn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullableVector<std::vector<int64_t>> s, nullableVector<std::vector<int64_t>> dim, std::string norm) {
  auto r_out = lantern_fft_ifftn_tensor_intarrayref_intarrayref_stdstring(self->get(), lantern_optional_vector_int64_t(s.x.data(), s.x.size(), s.is_null), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfftn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullableVector<std::vector<int64_t>> s, nullableVector<std::vector<int64_t>> dim, std::string norm) {
  auto r_out = lantern_fft_rfftn_tensor_intarrayref_intarrayref_stdstring(self->get(), lantern_optional_vector_int64_t(s.x.data(), s.x.size(), s.is_null), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfftn_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, nullableVector<std::vector<int64_t>> s, nullableVector<std::vector<int64_t>> dim, std::string norm) {
  auto r_out = lantern_fft_irfftn_tensor_intarrayref_intarrayref_stdstring(self->get(), lantern_optional_vector_int64_t(s.x.data(), s.x.size(), s.is_null), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), XPtrTorchstring(lantern_string_new(norm.c_str())).get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<XPtrTorchTensor> self, nullable<int64_t> signal_ndim, bool normalized) {
  auto r_out = lantern_fft_tensor_intt_bool(self->get(), signal_ndim.get(), reinterpret_cast<void*>(&normalized));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_det_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_linalg_det_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_det_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self) {
  auto r_out = lantern_det_tensor(self->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_outer_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_outer_tensor_tensor(self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_outer_out_out_Tensor_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_outer_out_tensor_tensor_tensor(out->get(), self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ger_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_ger_tensor_tensor(self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ger_out_out_Tensor_self_Tensor_vec2_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> vec2) {
  auto r_out = lantern_ger_out_tensor_tensor_tensor(out->get(), self->get(), vec2->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> ord, nullableVector<std::vector<int64_t>> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_linalg_norm_tensor_scalar_intarrayref_bool_scalartype(self->get(), ord->get(), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_self_Tensor_ord_stdstring (Rcpp::XPtr<XPtrTorchTensor> self, std::string ord, nullableVector<std::vector<int64_t>> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_linalg_norm_tensor_stdstring_intarrayref_bool_scalartype(self->get(), XPtrTorchstring(lantern_string_new(ord.c_str())).get(), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_out_out_Tensor_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchScalar> ord, nullableVector<std::vector<int64_t>> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_linalg_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out->get(), self->get(), ord->get(), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_out_out_Tensor_self_Tensor_ord_stdstring (Rcpp::XPtr<XPtrTorchTensor> out, Rcpp::XPtr<XPtrTorchTensor> self, std::string ord, nullableVector<std::vector<int64_t>> dim, bool keepdim, Rcpp::XPtr<XPtrTorch> dtype) {
  auto r_out = lantern_linalg_norm_out_tensor_tensor_stdstring_intarrayref_bool_scalartype(out->get(), self->get(), XPtrTorchstring(lantern_string_new(ord.c_str())).get(), lantern_optional_vector_int64_t(dim.x.data(), dim.x.size(), dim.is_null), reinterpret_cast<void*>(&keepdim), dtype->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_serialization_subcmul_self_Tensor_other_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> other, Rcpp::XPtr<XPtrTorchScalar> alpha) {
  auto r_out = lantern__test_serialization_subcmul_tensor_tensor_scalar(self->get(), other->get(), alpha->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_intlist_values_Tensor_addends_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> values, nullableVector<std::vector<int64_t>> addends) {
  auto r_out = lantern__test_optional_intlist_tensor_intarrayref(values->get(), lantern_optional_vector_int64_t(addends.x.data(), addends.x.size(), addends.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_filled_intlist_values_Tensor_addends_IntArrayRef (Rcpp::XPtr<XPtrTorchTensor> values, nullableVector<std::vector<int64_t>> addends) {
  auto r_out = lantern__test_optional_filled_intlist_tensor_intarrayref(values->get(), lantern_optional_vector_int64_t(addends.x.data(), addends.x.size(), addends.is_null));
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_floatlist_values_Tensor_addends_ArrayRefdouble (Rcpp::XPtr<XPtrTorchTensor> values, nullableVector<std::vector<double>> addends) {
  auto r_out = lantern__test_optional_floatlist_tensor_arrayrefdouble(values->get(), lantern_optional_vector_double(addends.x.data(), addends.x.size(), addends.is_null));
return XPtrTorchTensor(r_out);
}

