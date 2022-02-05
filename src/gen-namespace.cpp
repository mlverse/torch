// This file is auto generated. Dont modify it by hand.
#include <torch.h>

// [[Rcpp::export]]
void cpp_torch_method_set_data_self_Tensor_new_data_Tensor (XPtrTorchTensor self, XPtrTorchTensor new_data) {
  lantern_Tensor_set_data_tensor_tensor(self.get(), new_data.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_data_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_data_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_leaf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_leaf_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_output_nr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_output_nr_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method__version_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__version_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_requires_grad__self_Tensor (XPtrTorchTensor self, XPtrTorchbool requires_grad) {
  auto r_out = lantern_Tensor_requires_grad__tensor_bool(self.get(), requires_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_method_retain_grad_self_Tensor (XPtrTorchTensor self) {
  lantern_Tensor_retain_grad_tensor(self.get());
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_retains_grad_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_retains_grad_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__fw_primal_self_Tensor_level_int64_t (XPtrTorchTensor self, XPtrTorchint64_t level) {
  auto r_out = lantern_Tensor__fw_primal_tensor_intt(self.get(), level.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rename__self_Tensor_names_DimnameList (XPtrTorchTensor self, XPtrTorchOptionalDimnameList names) {
  auto r_out = lantern_Tensor_rename__tensor_dimnamelist(self.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rename_self_Tensor_names_DimnameList (XPtrTorchTensor self, XPtrTorchOptionalDimnameList names) {
  auto r_out = lantern_Tensor_rename_tensor_dimnamelist(self.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_to_self_Tensor_names_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList names) {
  auto r_out = lantern_Tensor_align_to_tensor_dimnamelist(self.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_to_self_Tensor_order_DimnameList_ellipsis_idx_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList order, XPtrTorchint64_t ellipsis_idx) {
  auto r_out = lantern_Tensor_align_to_tensor_dimnamelist_intt(self.get(), order.get(), ellipsis_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_align_as_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_align_as_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_refine_names_self_Tensor_names_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList names) {
  auto r_out = lantern_Tensor_refine_names_tensor_dimnamelist(self.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_abs_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_abs_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_abs__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_abs__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_absolute_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_absolute_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_absolute__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_absolute__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_angle_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_angle_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sgn_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sgn_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sgn__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sgn__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__conj_physical_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__conj_physical_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_conj_physical_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_conj_physical_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_conj_physical__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_conj_physical__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resolve_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_resolve_conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resolve_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_resolve_neg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__neg_view_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__neg_view_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_acos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_acos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arccos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arccos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_add_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_add__tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_add_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_add__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_add__tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmv_self_Tensor_mat_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat, XPtrTorchTensor vec, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addmv_tensor_tensor_tensor_scalar_scalar(self.get(), mat.get(), vec.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmv__self_Tensor_mat_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat, XPtrTorchTensor vec, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addmv__tensor_tensor_tensor_scalar_scalar(self.get(), mat.get(), vec.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addr_self_Tensor_vec1_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec1, XPtrTorchTensor vec2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addr_tensor_tensor_tensor_scalar_scalar(self.get(), vec1.get(), vec2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addr__self_Tensor_vec1_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec1, XPtrTorchTensor vec2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addr__tensor_tensor_tensor_scalar_scalar(self.get(), vec1.get(), vec2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_all_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_all_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_allclose_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchdouble rtol, XPtrTorchdouble atol, XPtrTorchbool equal_nan) {
  auto r_out = lantern_Tensor_allclose_tensor_tensor_double_double_bool(self.get(), other.get(), rtol.get(), atol.get(), equal_nan.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_any_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_any_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argmax_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_argmax_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argmin_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_argmin_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_acosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_acosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_acosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arccosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arccosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arccosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_asinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_asinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arcsinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arcsinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_atanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_atanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arctanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arctanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchoptional_int64_t storage_offset) {
  auto r_out = lantern_Tensor_as_strided_tensor_intarrayref_intarrayref_intt(self.get(), size.get(), stride.get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchoptional_int64_t storage_offset) {
  auto r_out = lantern_Tensor_as_strided__tensor_intarrayref_intarrayref_intt(self.get(), size.get(), stride.get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_asin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_asin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_asin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arcsin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arcsin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arcsin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_atan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_atan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arctan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_arctan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_arctan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_baddbmm_tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_baddbmm__self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_baddbmm__tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_bernoulli_tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli__self_Tensor_p_Tensor (XPtrTorchTensor self, XPtrTorchTensor p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_bernoulli__tensor_tensor_generator(self.get(), p.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli__self_Tensor_p_double (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_bernoulli__tensor_double_generator(self.get(), p.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bernoulli_self_Tensor_p_double (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_bernoulli_tensor_double_generator(self.get(), p.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bincount_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor weights, XPtrTorchint64_t minlength) {
  auto r_out = lantern_Tensor_bincount_tensor_tensor_intt(self.get(), weights.get(), minlength.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_not_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_bitwise_not_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_not__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_bitwise_not__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copysign_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_copysign_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copysign__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_copysign__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copysign_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_copysign_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copysign__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_copysign__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_not_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_logical_not_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_not__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_logical_not__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_xor_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_xor_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_xor__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_xor__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_and_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_and_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_and__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_and__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_or_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_or_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logical_or__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logical_or__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bmm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_Tensor_bmm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_broadcast_to_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_Tensor_broadcast_to_tensor_intarrayref(self.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ceil_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_ceil_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ceil__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_ceil__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_unsafe_chunk_self_Tensor_chunks_int64_t (XPtrTorchTensor self, XPtrTorchint64_t chunks, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unsafe_chunk_tensor_intt_intt(self.get(), chunks.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_chunk_self_Tensor_chunks_int64_t (XPtrTorchTensor self, XPtrTorchint64_t chunks, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_chunk_tensor_intt_intt(self.get(), chunks.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_tensor_split_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_tensor_split_tensor_intt_intt(self.get(), sections.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_tensor_split_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_tensor_split_tensor_intarrayref_intt(self.get(), indices.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_tensor_split_self_Tensor_tensor_indices_or_sections_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor_indices_or_sections, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_tensor_split_tensor_tensor_intt(self.get(), tensor_indices_or_sections.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_Tensor_clamp_tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_Tensor_clamp_tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp__self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_Tensor_clamp__tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp__self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_Tensor_clamp__tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max_self_Tensor_max_Scalar (XPtrTorchTensor self, XPtrTorchScalar max) {
  auto r_out = lantern_Tensor_clamp_max_tensor_scalar(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max_self_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchTensor max) {
  auto r_out = lantern_Tensor_clamp_max_tensor_tensor(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max__self_Tensor_max_Scalar (XPtrTorchTensor self, XPtrTorchScalar max) {
  auto r_out = lantern_Tensor_clamp_max__tensor_scalar(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_max__self_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchTensor max) {
  auto r_out = lantern_Tensor_clamp_max__tensor_tensor(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min_self_Tensor_min_Scalar (XPtrTorchTensor self, XPtrTorchScalar min) {
  auto r_out = lantern_Tensor_clamp_min_tensor_scalar(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min_self_Tensor_min_Tensor (XPtrTorchTensor self, XPtrTorchTensor min) {
  auto r_out = lantern_Tensor_clamp_min_tensor_tensor(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min__self_Tensor_min_Scalar (XPtrTorchTensor self, XPtrTorchScalar min) {
  auto r_out = lantern_Tensor_clamp_min__tensor_scalar(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clamp_min__self_Tensor_min_Tensor (XPtrTorchTensor self, XPtrTorchTensor min) {
  auto r_out = lantern_Tensor_clamp_min__tensor_tensor(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_Tensor_clip_tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_Tensor_clip_tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip__self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_Tensor_clip__tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clip__self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_Tensor_clip__tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_contiguous_self_Tensor (XPtrTorchTensor self, XPtrTorchMemoryFormat memory_format) {
  auto r_out = lantern_Tensor_contiguous_tensor_memoryformat(self.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_copy__self_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchTensor src, XPtrTorchbool non_blocking) {
  auto r_out = lantern_Tensor_copy__tensor_tensor_bool(self.get(), src.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_cos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_cos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_cosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_cosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_count_nonzero_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim) {
  auto r_out = lantern_Tensor_count_nonzero_tensor_intarrayref(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_count_nonzero_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_Tensor_count_nonzero_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cov_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t correction, XPtrTorchOptionalTensor fweights, XPtrTorchOptionalTensor aweights) {
  auto r_out = lantern_Tensor_cov_tensor_intt_tensor_tensor(self.get(), correction.get(), fweights.get(), aweights.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_corrcoef_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_corrcoef_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_cummax_tensor_intt(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_cummax_tensor_dimname(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummin_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_cummin_tensor_intt(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_cummin_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_cummin_tensor_dimname(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumprod_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod__self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumprod__tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumprod_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumprod__self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumprod__tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumsum_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum__self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumsum__tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumsum_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cumsum__self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_cumsum__tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diag_embed_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t offset, XPtrTorchindex_int64_t dim1, XPtrTorchindex_int64_t dim2) {
  auto r_out = lantern_Tensor_diag_embed_tensor_intt_intt_intt(self.get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagflat_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t offset) {
  auto r_out = lantern_Tensor_diagflat_tensor_intt(self.get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagonal_self_Tensor_dim1_int64_t_dim2_int64_t (XPtrTorchTensor self, XPtrTorchint64_t offset, XPtrTorchindex_int64_t dim1, XPtrTorchindex_int64_t dim2) {
  auto r_out = lantern_Tensor_diagonal_tensor_intt_intt_intt(self.get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diagonal_self_Tensor_outdim_Dimname_dim1_Dimname_dim2_Dimname (XPtrTorchTensor self, XPtrTorchDimname outdim, XPtrTorchDimname dim1, XPtrTorchDimname dim2, XPtrTorchint64_t offset) {
  auto r_out = lantern_Tensor_diagonal_tensor_dimname_dimname_dimname_intt(self.get(), outdim.get(), dim1.get(), dim2.get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill_diagonal__self_Tensor_fill_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar fill_value, XPtrTorchbool wrap) {
  auto r_out = lantern_Tensor_fill_diagonal__tensor_scalar_bool(self.get(), fill_value.get(), wrap.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diff_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t n, XPtrTorchindex_int64_t dim, XPtrTorchOptionalTensor prepend, XPtrTorchOptionalTensor append) {
  auto r_out = lantern_Tensor_diff_tensor_intt_intt_tensor_tensor(self.get(), n.get(), dim.get(), prepend.get(), append.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_div_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_div__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_div_tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_div__tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_div_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_div__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div_self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_div_tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_div__self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_div__tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_divide__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_divide__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_divide_tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_divide__tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide_self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_divide_tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_divide__self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_Tensor_divide__tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_true_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_true_divide__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_true_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_true_divide__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_true_divide__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dot_self_Tensor_tensor_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor) {
  auto r_out = lantern_Tensor_dot_tensor_tensor(self.get(), tensor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_vdot_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_vdot_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_empty_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_Tensor_new_empty_tensor_intarrayref_tensoroptions(self.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_empty_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchTensorOptions options) {
  auto r_out = lantern_Tensor_new_empty_strided_tensor_intarrayref_intarrayref_tensoroptions(self.get(), size.get(), stride.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_full_self_Tensor_size_IntArrayRef_fill_value_Scalar (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchScalar fill_value, XPtrTorchTensorOptions options) {
  auto r_out = lantern_Tensor_new_full_tensor_intarrayref_scalar_tensoroptions(self.get(), size.get(), fill_value.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_zeros_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_Tensor_new_zeros_tensor_intarrayref_tensoroptions(self.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_new_ones_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_Tensor_new_ones_tensor_intarrayref_tensoroptions(self.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resize__self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_resize__tensor_intarrayref_memoryformat(self.get(), size.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erf__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erf__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erfc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erfc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_exp_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_exp__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp2_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_exp2_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exp2__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_exp2__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expm1_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_expm1_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expm1__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_expm1__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expand_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchbool implicit) {
  auto r_out = lantern_Tensor_expand_tensor_intarrayref_bool(self.get(), size.get(), implicit.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_expand_as_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_expand_as_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t start_dim, XPtrTorchindex_int64_t end_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_intt_intt(self.get(), start_dim.get(), end_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchindex_int64_t start_dim, XPtrTorchindex_int64_t end_dim, XPtrTorchDimname out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_intt_intt_dimname(self.get(), start_dim.get(), end_dim.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname start_dim, XPtrTorchDimname end_dim, XPtrTorchDimname out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_dimname_dimname_dimname(self.get(), start_dim.get(), end_dim.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimnameList dims, XPtrTorchDimname out_dim) {
  auto r_out = lantern_Tensor_flatten_tensor_dimnamelist_dimname(self.get(), dims.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unflatten_self_Tensor_dim_int64_t_sizes_IntArrayRef_names_DimnameList (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIntArrayRef sizes, XPtrTorchOptionalDimnameList names) {
  auto r_out = lantern_Tensor_unflatten_tensor_intt_intarrayref_dimnamelist(self.get(), dim.get(), sizes.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unflatten_self_Tensor_dim_Dimname_sizes_IntArrayRef_names_DimnameList (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIntArrayRef sizes, XPtrTorchDimnameList names) {
  auto r_out = lantern_Tensor_unflatten_tensor_dimname_intarrayref_dimnamelist(self.get(), dim.get(), sizes.get(), names.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill__self_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_fill__tensor_scalar(self.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fill__self_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_fill__tensor_tensor(self.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_floor_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_floor__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_floor_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_floor_divide__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_floor_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_floor_divide__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_floor_divide__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_frac_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_frac_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_frac__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_frac__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gcd_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_gcd_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gcd__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_gcd__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lcm_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_lcm_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lcm__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_lcm__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_self_Tensor_indices_constc10Listc10optionalTensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices) {
  auto r_out = lantern_Tensor_index_tensor_constclistcoptionaltensor(self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_copy__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_copy_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy__self_Tensor_dim_Dimname_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_copy__tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_copy_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_put__self_Tensor_indices_constc10Listc10optionalTensor_values_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices, XPtrTorchTensor values, XPtrTorchbool accumulate) {
  auto r_out = lantern_Tensor_index_put__tensor_constclistcoptionaltensor_tensor_bool(self.get(), indices.get(), values.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_put_self_Tensor_indices_constc10Listc10optionalTensor_values_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices, XPtrTorchTensor values, XPtrTorchbool accumulate) {
  auto r_out = lantern_Tensor_index_put_tensor_constclistcoptionaltensor_tensor_bool(self.get(), indices.get(), values.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_inverse_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_inverse_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isclose_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchdouble rtol, XPtrTorchdouble atol, XPtrTorchbool equal_nan) {
  auto r_out = lantern_Tensor_isclose_tensor_tensor_double_double_bool(self.get(), other.get(), rtol.get(), atol.get(), equal_nan.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isnan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isnan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_distributed_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_distributed_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_floating_point_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_floating_point_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_complex_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_complex_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_conj_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_neg_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isreal_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isreal_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_nonzero_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_nonzero_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_same_size_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_is_same_size_tensor_tensor(self.get(), other.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_signed_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_signed_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_inference_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_inference_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_kron_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_kron_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t_dim_int64_t (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_kthvalue_tensor_intt_intt_bool(self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t_dim_Dimname (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_kthvalue_tensor_intt_dimname_bool(self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nan_to_num_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble nan, XPtrTorchOptionaldouble posinf, XPtrTorchOptionaldouble neginf) {
  auto r_out = lantern_Tensor_nan_to_num_tensor_double_double_double(self.get(), nan.get(), posinf.get(), neginf.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nan_to_num__self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble nan, XPtrTorchOptionaldouble posinf, XPtrTorchOptionaldouble neginf) {
  auto r_out = lantern_Tensor_nan_to_num__tensor_double_double_double(self.get(), nan.get(), posinf.get(), neginf.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ldexp_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ldexp_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ldexp__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ldexp__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log10_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log10_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log10__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log10__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log1p_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log1p_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log1p__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log1p__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log2_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log2_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log2__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_log2__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logaddexp_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logaddexp_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logaddexp2_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_logaddexp2_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_xlogy_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_xlogy_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_xlogy_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_xlogy_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_xlogy__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_xlogy__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_xlogy__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_xlogy__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logdet_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_logdet_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_log_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_log_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logcumsumexp_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_logcumsumexp_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logcumsumexp_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_logcumsumexp_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logsumexp_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_logsumexp_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logsumexp_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_logsumexp_tensor_dimnamelist_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matmul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_matmul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matrix_power_self_Tensor_n_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_Tensor_matrix_power_tensor_intt(self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_matrix_exp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_matrix_exp_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_aminmax_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_aminmax_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_max_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_max_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_amax_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_amax_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_mean_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_mean_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mean_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_mean_tensor_dimnamelist_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanmean_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_nanmean_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_median_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_median_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_median_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_median_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanmedian_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_nanmedian_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_nanmedian_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_nanmedian_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_nanmedian_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_nanmedian_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_min_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_min_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_amin_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_amin_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_Tensor_mm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_mode_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_mode_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_mul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_mul__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_mul_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mul__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_mul__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_multiply_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_multiply__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_multiply_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multiply__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_multiply__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mv_self_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec) {
  auto r_out = lantern_Tensor_mv_tensor_tensor(self.get(), vec.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mvlgamma_self_Tensor_p_int64_t (XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_Tensor_mvlgamma_tensor_intt(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_mvlgamma__self_Tensor_p_int64_t (XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_Tensor_mvlgamma__tensor_intt(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_copy_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t length) {
  auto r_out = lantern_Tensor_narrow_copy_tensor_intt_intt_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t length) {
  auto r_out = lantern_Tensor_narrow_tensor_intt_intt_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_narrow_self_Tensor_dim_int64_t_start_Tensor_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchTensor start, XPtrTorchint64_t length) {
  auto r_out = lantern_Tensor_narrow_tensor_intt_tensor_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_permute_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_Tensor_permute_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_movedim_self_Tensor_source_IntArrayRef_destination_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef source, XPtrTorchIntArrayRef destination) {
  auto r_out = lantern_Tensor_movedim_tensor_intarrayref_intarrayref(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_movedim_self_Tensor_source_int64_t_destination_int64_t (XPtrTorchTensor self, XPtrTorchint64_t source, XPtrTorchint64_t destination) {
  auto r_out = lantern_Tensor_movedim_tensor_intt_intt(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_moveaxis_self_Tensor_source_IntArrayRef_destination_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef source, XPtrTorchIntArrayRef destination) {
  auto r_out = lantern_Tensor_moveaxis_tensor_intarrayref_intarrayref(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_moveaxis_self_Tensor_source_int64_t_destination_int64_t (XPtrTorchTensor self, XPtrTorchint64_t source, XPtrTorchint64_t destination) {
  auto r_out = lantern_Tensor_moveaxis_tensor_intt_intt(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_numpy_T_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_numpy_t_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_pinned_self_Tensor (XPtrTorchTensor self, XPtrTorchDevice device) {
  auto r_out = lantern_Tensor_is_pinned_tensor_device(self.get(), device.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pin_memory_self_Tensor (XPtrTorchTensor self, XPtrTorchDevice device) {
  auto r_out = lantern_Tensor_pin_memory_tensor_device(self.get(), device.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pinverse_self_Tensor (XPtrTorchTensor self, XPtrTorchdouble rcond) {
  auto r_out = lantern_Tensor_pinverse_tensor_double(self.get(), rcond.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rad2deg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_rad2deg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rad2deg__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_rad2deg__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_deg2rad_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_deg2rad_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_deg2rad__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_deg2rad__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ravel_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_ravel_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reciprocal_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_reciprocal_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reciprocal__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_reciprocal__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_neg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_neg__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_neg__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_negative_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_negative_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_negative__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_negative__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_self_Tensor_repeats_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef repeats) {
  auto r_out = lantern_Tensor_repeat_tensor_intarrayref(self.get(), repeats.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_interleave_self_Tensor_repeats_Tensor (XPtrTorchTensor self, XPtrTorchTensor repeats, XPtrTorchoptional_index_int64_t dim, XPtrTorchoptional_int64_t output_size) {
  auto r_out = lantern_Tensor_repeat_interleave_tensor_tensor_intt_intt(self.get(), repeats.get(), dim.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_repeat_interleave_self_Tensor_repeats_int64_t (XPtrTorchTensor self, XPtrTorchint64_t repeats, XPtrTorchoptional_index_int64_t dim, XPtrTorchoptional_int64_t output_size) {
  auto r_out = lantern_Tensor_repeat_interleave_tensor_intt_intt_intt(self.get(), repeats.get(), dim.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reshape_self_Tensor_shape_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef shape) {
  auto r_out = lantern_Tensor_reshape_tensor_intarrayref(self.get(), shape.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__reshape_alias_self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_Tensor__reshape_alias_tensor_intarrayref_intarrayref(self.get(), size.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_reshape_as_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_reshape_as_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_round_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_round_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_round__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_round__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_relu_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_relu_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_relu__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_relu__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prelu_self_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor weight) {
  auto r_out = lantern_Tensor_prelu_tensor_tensor(self.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight) {
  auto r_out = lantern_Tensor_prelu_backward_tensor_tensor_tensor(grad_output.get(), self.get(), weight.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hardshrink_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_Tensor_hardshrink_tensor_scalar(self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (XPtrTorchTensor grad_out, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_Tensor_hardshrink_backward_tensor_tensor_scalar(grad_out.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rsqrt_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_rsqrt_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rsqrt__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_rsqrt__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_select_self_Tensor_dim_Dimname_index_int64_t (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchindex_int64_t index) {
  auto r_out = lantern_Tensor_select_tensor_dimname_intt(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_select_self_Tensor_dim_int64_t_index_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchindex_int64_t index) {
  auto r_out = lantern_Tensor_select_tensor_intt_intt(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sigmoid_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sigmoid_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sigmoid__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sigmoid__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logit_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_Tensor_logit_tensor_double(self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_logit__self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_Tensor_logit__tensor_double(self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sinc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sinc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_detach_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_detach_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_detach__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_detach__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_size_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_size_tensor_dimname(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_slice_self_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_int64_t start, XPtrTorchoptional_int64_t end, XPtrTorchint64_t step) {
  auto r_out = lantern_Tensor_slice_tensor_intt_intt_intt_intt(self.get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_slogdet_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_slogdet_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_smm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_Tensor_smm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_unsafe_split_self_Tensor_split_size_int64_t (XPtrTorchTensor self, XPtrTorchint64_t split_size, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unsafe_split_tensor_intt_intt(self.get(), split_size.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_split_self_Tensor_split_size_int64_t (XPtrTorchTensor self, XPtrTorchint64_t split_size, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_split_tensor_intt_intt(self.get(), split_size.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_unsafe_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef split_sizes, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unsafe_split_with_sizes_tensor_intarrayref_intt(self.get(), split_sizes.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef split_sizes, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_split_with_sizes_tensor_intarrayref_intt(self.get(), split_sizes.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_hsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_Tensor_hsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_hsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_Tensor_hsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_vsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_Tensor_vsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_vsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_Tensor_vsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_dsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_Tensor_dsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_dsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_Tensor_dsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_squeeze_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_squeeze_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_squeeze_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_squeeze__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_squeeze__tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_squeeze__self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_squeeze__tensor_dimname(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_sspaddmm_tensor_tensor_tensor_scalar_scalar(self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_stft_self_Tensor_n_fft_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n_fft, XPtrTorchoptional_int64_t hop_length, XPtrTorchoptional_int64_t win_length, XPtrTorchOptionalTensor window, XPtrTorchbool normalized, XPtrTorchoptional_bool onesided, XPtrTorchoptional_bool return_complex) {
  auto r_out = lantern_Tensor_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(self.get(), n_fft.get(), hop_length.get(), win_length.get(), window.get(), normalized.get(), onesided.get(), return_complex.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_istft_self_Tensor_n_fft_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n_fft, XPtrTorchoptional_int64_t hop_length, XPtrTorchoptional_int64_t win_length, XPtrTorchOptionalTensor window, XPtrTorchbool center, XPtrTorchbool normalized, XPtrTorchoptional_bool onesided, XPtrTorchoptional_int64_t length, XPtrTorchbool return_complex) {
  auto r_out = lantern_Tensor_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(self.get(), n_fft.get(), hop_length.get(), win_length.get(), window.get(), center.get(), normalized.get(), onesided.get(), length.get(), return_complex.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_stride_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_stride_tensor_intt(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_stride_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_stride_tensor_dimname(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_sum_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_sum_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_sum_tensor_dimnamelist_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nansum_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_nansum_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nansum_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_nansum_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sum_to_size_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_Tensor_sum_to_size_tensor_intarrayref(self.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sqrt_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sqrt_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sqrt__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sqrt__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_square_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_square_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_square__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_square__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_Tensor_std_tensor_bool(self.get(), unbiased.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_std_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_std_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_prod_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_prod_tensor_intt_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_prod_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_prod_tensor_dimname_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_t_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_t_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_t__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_t__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_tan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_tan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_tanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_tanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tile_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_Tensor_tile_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_Tensor_transpose_tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim0, XPtrTorchDimname dim1) {
  auto r_out = lantern_Tensor_transpose_tensor_dimname_dimname(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_Tensor_transpose__tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flip_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_Tensor_flip_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fliplr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_fliplr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_flipud_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_flipud_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_roll_self_Tensor_shifts_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef shifts, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_Tensor_roll_tensor_intarrayref_intarrayref(self.get(), shifts.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_rot90_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_Tensor_rot90_tensor_intt_intarrayref(self.get(), k.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trunc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_trunc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trunc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_trunc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fix_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_fix_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fix__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_fix__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_type_as_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_type_as_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unsqueeze_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unsqueeze_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unsqueeze__self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unsqueeze__tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_Tensor_var_tensor_bool(self.get(), unbiased.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_var_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_var_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_view_as_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_view_as_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_where_condition_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor condition, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_where_tensor_tensor_tensor(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDtype dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_scalartype(self.get(), p.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar (XPtrTorchTensor self, XPtrTorchScalar p) {
  auto r_out = lantern_Tensor_norm_tensor_scalar(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_intarrayref_bool(self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_dimnamelist_bool_scalartype(self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_norm_tensor_scalar_dimnamelist_bool(self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_frexp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_frexp_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_clone_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_clone_tensor_memoryformat(self.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_positive_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_positive_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_resize_as__self_Tensor_the_template_Tensor (XPtrTorchTensor self, XPtrTorchTensor the_template, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_resize_as__tensor_tensor_memoryformat(self.get(), the_template.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_zero__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_zero__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_sub_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_sub__tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_sub_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sub__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_sub__tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_subtract_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_subtract__tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_subtract_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_subtract__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_subtract__tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_heaviside_self_Tensor_values_Tensor (XPtrTorchTensor self, XPtrTorchTensor values) {
  auto r_out = lantern_Tensor_heaviside_tensor_tensor(self.get(), values.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_heaviside__self_Tensor_values_Tensor (XPtrTorchTensor self, XPtrTorchTensor values) {
  auto r_out = lantern_Tensor_heaviside__tensor_tensor(self.get(), values.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addmm_tensor_tensor_tensor_scalar_scalar(self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addmm__self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addmm__tensor_tensor_tensor_scalar_scalar(self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_resize__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchint64_t sparse_dim, XPtrTorchint64_t dense_dim) {
  auto r_out = lantern_Tensor_sparse_resize__tensor_intarrayref_intt_intt(self.get(), size.get(), sparse_dim.get(), dense_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_resize_and_clear__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchint64_t sparse_dim, XPtrTorchint64_t dense_dim) {
  auto r_out = lantern_Tensor_sparse_resize_and_clear__tensor_intarrayref_intt_intt(self.get(), size.get(), sparse_dim.get(), dense_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sparse_mask_self_Tensor_mask_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask) {
  auto r_out = lantern_Tensor_sparse_mask_tensor_tensor(self.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_dense_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_to_dense_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_sparse_dim_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sparse_dim_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method__dimI_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__dimi_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_dense_dim_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_dense_dim_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method__dimV_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__dimv_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method__nnz_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__nnz_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_coalesce_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_coalesce_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_coalesced_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_is_coalesced_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__indices_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__indices_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__values_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor__values_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method__coalesced__self_Tensor_coalesced_bool (XPtrTorchTensor self, XPtrTorchbool coalesced) {
  auto r_out = lantern_Tensor__coalesced__tensor_bool(self.get(), coalesced.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_indices_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_indices_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_values_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_values_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_crow_indices_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_crow_indices_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_col_indices_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_col_indices_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_unbind_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_Tensor_unbind_tensor_intt(self.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_unbind_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_Tensor_unbind_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_sparse_self_Tensor_sparse_dim_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sparse_dim) {
  auto r_out = lantern_Tensor_to_sparse_tensor_intt(self.get(), sparse_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_sparse_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_to_sparse_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_mkldnn_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_Tensor_to_mkldnn_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dequantize_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_dequantize_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchdouble cpp_torch_method_q_scale_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_q_scale_tensor(self.get());
return XPtrTorchdouble(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_q_zero_point_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_q_zero_point_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_q_per_channel_scales_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_q_per_channel_scales_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_q_per_channel_zero_points_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_q_per_channel_zero_points_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_method_q_per_channel_axis_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_q_per_channel_axis_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_int_repr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_int_repr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_method_qscheme_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_qscheme_tensor(self.get());
return make_xptr<XPtrTorchQScheme>(r_out, "QScheme");
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchbool non_blocking, XPtrTorchbool copy, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_to_tensor_tensoroptions_bool_bool_memoryformat(self.get(), options.get(), non_blocking.get(), copy.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_device_Device_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchDevice device, XPtrTorchDtype dtype, XPtrTorchbool non_blocking, XPtrTorchbool copy, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_to_tensor_device_scalartype_bool_bool_memoryformat(self.get(), device.get(), dtype.get(), non_blocking.get(), copy.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchDtype dtype, XPtrTorchbool non_blocking, XPtrTorchbool copy, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_to_tensor_scalartype_bool_bool_memoryformat(self.get(), dtype.get(), non_blocking.get(), copy.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_to_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchbool non_blocking, XPtrTorchbool copy, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_Tensor_to_tensor_tensor_bool_bool_memoryformat(self.get(), other.get(), non_blocking.get(), copy.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalar cpp_torch_method_item_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_item_tensor(self.get());
return XPtrTorchScalar(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Storage (XPtrTorchTensor self, Rcpp::XPtr<XPtrTorch> source) {
  auto r_out = lantern_Tensor_set__tensor_storage(self.get(), source->get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Storage_storage_offset_int64_t_size_IntArrayRef (XPtrTorchTensor self, Rcpp::XPtr<XPtrTorch> source, XPtrTorchint64_t storage_offset, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_Tensor_set__tensor_storage_intt_intarrayref_intarrayref(self.get(), source->get(), storage_offset.get(), size.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_set__tensor_tensor(self.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_set__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_set__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_is_set_to_self_Tensor_tensor_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor) {
  auto r_out = lantern_Tensor_is_set_to_tensor_tensor(self.get(), tensor.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_masked_fill__tensor_tensor_scalar(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_masked_fill_tensor_tensor_scalar(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_masked_fill__tensor_tensor_tensor(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_masked_fill_tensor_tensor_tensor(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_scatter__self_Tensor_mask_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_masked_scatter__tensor_tensor_tensor(self.get(), mask.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_masked_scatter_tensor_tensor_tensor(self.get(), mask.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_view_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_Tensor_view_tensor_intarrayref(self.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_view_self_Tensor_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchDtype dtype) {
  auto r_out = lantern_Tensor_view_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_put__self_Tensor_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchbool accumulate) {
  auto r_out = lantern_Tensor_put__tensor_tensor_tensor_bool(self.get(), index.get(), source.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_put_self_Tensor_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchbool accumulate) {
  auto r_out = lantern_Tensor_put_tensor_tensor_tensor_bool(self.get(), index.get(), source.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_add__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add__self_Tensor_dim_int64_t_index_Tensor_source_Tensor_alpha_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_index_add__tensor_intt_tensor_tensor_scalar(self.get(), dim.get(), index.get(), source.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_Tensor_index_add_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor_alpha_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_index_add_tensor_intt_tensor_tensor_scalar(self.get(), dim.get(), index.get(), source.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor_alpha_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_index_add_tensor_dimname_tensor_tensor_scalar(self.get(), dim.get(), index.get(), source.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_index_fill__tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_index_fill_tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_index_fill__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_index_fill_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_index_fill__tensor_dimname_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_index_fill__tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_index_fill_tensor_dimname_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_Tensor_index_fill_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src, XPtrTorchstring_view reduce) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_tensor_cstringview(self.get(), dim.get(), index.get(), src.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_src_Tensor_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src, XPtrTorchstring_view reduce) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_tensor_cstringview(self.get(), dim.get(), index.get(), src.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value, XPtrTorchstring_view reduce) {
  auto r_out = lantern_Tensor_scatter_tensor_intt_tensor_scalar_cstringview(self.get(), dim.get(), index.get(), value.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_value_Scalar_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value, XPtrTorchstring_view reduce) {
  auto r_out = lantern_Tensor_scatter__tensor_intt_tensor_scalar_cstringview(self.get(), dim.get(), index.get(), value.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_scatter_tensor_dimname_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter_add_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter_add__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_Tensor_scatter_add_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_eq__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_eq__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_and_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_and_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_and__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_and__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_and__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___and___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___and___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___and___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___and___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___iand___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___iand___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___iand___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___iand___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_or_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_or_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_or__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_or__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_or__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___or___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___or___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___or___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___or___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ior___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___ior___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ior___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___ior___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_xor_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_xor_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_xor__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_xor__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_xor__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___xor___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___xor___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___xor___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___xor___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ixor___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___ixor___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ixor___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___ixor___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___lshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___lshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___lshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___lshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ilshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___ilshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___ilshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___ilshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_left_shift_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_left_shift_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_left_shift__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_left_shift__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_left_shift_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_left_shift_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_left_shift__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_left_shift__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___rshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___rshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___rshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___rshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___irshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor___irshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method___irshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor___irshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_right_shift_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_right_shift_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_right_shift__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_bitwise_right_shift__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_right_shift_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_right_shift_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_bitwise_right_shift__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_bitwise_right_shift__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tril__self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_Tensor_tril__tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_triu__self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_Tensor_triu__tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_digamma__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_digamma__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Scalar (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchScalar weight) {
  auto r_out = lantern_Tensor_lerp__tensor_tensor_scalar(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchTensor weight) {
  auto r_out = lantern_Tensor_lerp__tensor_tensor_tensor(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addbmm__self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addbmm__tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_Tensor_addbmm_tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor_from_int64_t_to_int64_t (XPtrTorchTensor self, XPtrTorchint64_t from, XPtrTorchoptional_int64_t to, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_random__tensor_intt_intt_generator(self.get(), from.get(), to.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor_to_int64_t (XPtrTorchTensor self, XPtrTorchint64_t to, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_random__tensor_intt_generator(self.get(), to.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_random__self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_random__tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_uniform__self_Tensor (XPtrTorchTensor self, XPtrTorchdouble from, XPtrTorchdouble to, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_uniform__tensor_double_double_generator(self.get(), from.get(), to.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cauchy__self_Tensor (XPtrTorchTensor self, XPtrTorchdouble median, XPtrTorchdouble sigma, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_cauchy__tensor_double_double_generator(self.get(), median.get(), sigma.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_log_normal__self_Tensor (XPtrTorchTensor self, XPtrTorchdouble mean, XPtrTorchdouble std, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_log_normal__tensor_double_double_generator(self.get(), mean.get(), std.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_exponential__self_Tensor (XPtrTorchTensor self, XPtrTorchdouble lambd, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_exponential__tensor_double_generator(self.get(), lambd.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_geometric__self_Tensor_p_double (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_geometric__tensor_double_generator(self.get(), p.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_diag_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_Tensor_diag_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cross_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_Tensor_cross_tensor_tensor_intt(self.get(), other.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_triu_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_Tensor_triu_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_tril_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_Tensor_tril_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_trace_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_trace_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_ne_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ne_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_ne__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ne__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ne__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_not_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_not_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_not_equal__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_not_equal__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_not_equal__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_eq_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_eq_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_eq_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_ge_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ge_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_ge__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ge__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_ge__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_greater_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_greater_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_greater_equal__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_equal__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_greater_equal__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_le_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_le_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_le__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_le__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_le__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_less_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_less_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_less_equal__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_equal__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_less_equal__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_gt_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_gt_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_gt__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gt__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_gt__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_greater_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_greater_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_greater__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_greater__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_greater__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_lt_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_lt_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_lt__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lt__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_lt__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_less_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_less_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_less__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_less__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_less__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_take_self_Tensor_index_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor index) {
  auto r_out = lantern_Tensor_take_tensor_tensor(self.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_take_along_dim_self_Tensor_indices_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_Tensor_take_along_dim_tensor_tensor_intt(self.get(), indices.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_select_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_Tensor_index_select_tensor_intt_tensor(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_index_select_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_Tensor_index_select_tensor_dimname_tensor(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_masked_select_self_Tensor_mask_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask) {
  auto r_out = lantern_Tensor_masked_select_tensor_tensor(self.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nonzero_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_nonzero_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_method_nonzero_numpy_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_nonzero_numpy_tensor(self.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gather_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_Tensor_gather_tensor_intt_tensor_bool(self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_gather_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_Tensor_gather_tensor_dimname_tensor_bool(self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_addcmul_tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcmul__self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_addcmul__tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_addcdiv_tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_addcdiv__self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_Tensor_addcdiv__tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_lstsq_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_Tensor_lstsq_tensor_tensor(self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_triangular_solve_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A, XPtrTorchbool upper, XPtrTorchbool transpose, XPtrTorchbool unitriangular) {
  auto r_out = lantern_Tensor_triangular_solve_tensor_tensor_bool_bool_bool(self.get(), A.get(), upper.get(), transpose.get(), unitriangular.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_symeig_self_Tensor (XPtrTorchTensor self, XPtrTorchbool eigenvectors, XPtrTorchbool upper) {
  auto r_out = lantern_Tensor_symeig_tensor_bool_bool(self.get(), eigenvectors.get(), upper.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_eig_self_Tensor (XPtrTorchTensor self, XPtrTorchbool eigenvectors) {
  auto r_out = lantern_Tensor_eig_tensor_bool(self.get(), eigenvectors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_svd_self_Tensor (XPtrTorchTensor self, XPtrTorchbool some, XPtrTorchbool compute_uv) {
  auto r_out = lantern_Tensor_svd_tensor_bool_bool(self.get(), some.get(), compute_uv.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_swapaxes_self_Tensor_axis0_int64_t_axis1_int64_t (XPtrTorchTensor self, XPtrTorchint64_t axis0, XPtrTorchint64_t axis1) {
  auto r_out = lantern_Tensor_swapaxes_tensor_intt_intt(self.get(), axis0.get(), axis1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_swapaxes__self_Tensor_axis0_int64_t_axis1_int64_t (XPtrTorchTensor self, XPtrTorchint64_t axis0, XPtrTorchint64_t axis1) {
  auto r_out = lantern_Tensor_swapaxes__tensor_intt_intt(self.get(), axis0.get(), axis1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_swapdims_self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_Tensor_swapdims_tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_swapdims__self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_Tensor_swapdims__tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_Tensor_cholesky_tensor_bool(self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_solve_self_Tensor_input2_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchbool upper) {
  auto r_out = lantern_Tensor_cholesky_solve_tensor_tensor_bool(self.get(), input2.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_solve_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_Tensor_solve_tensor_tensor(self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_cholesky_inverse_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_Tensor_cholesky_inverse_tensor_bool(self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_qr_self_Tensor (XPtrTorchTensor self, XPtrTorchbool some) {
  auto r_out = lantern_Tensor_qr_tensor_bool(self.get(), some.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_geqrf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_geqrf_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_orgqr_self_Tensor_input2_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2) {
  auto r_out = lantern_Tensor_orgqr_tensor_tensor(self.get(), input2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ormqr_self_Tensor_input2_Tensor_input3_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchTensor input3, XPtrTorchbool left, XPtrTorchbool transpose) {
  auto r_out = lantern_Tensor_ormqr_tensor_tensor_tensor_bool_bool(self.get(), input2.get(), input3.get(), left.get(), transpose.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (XPtrTorchTensor self, XPtrTorchTensor LU_data, XPtrTorchTensor LU_pivots) {
  auto r_out = lantern_Tensor_lu_solve_tensor_tensor_tensor(self.get(), LU_data.get(), LU_pivots.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_multinomial_self_Tensor_num_samples_int64_t (XPtrTorchTensor self, XPtrTorchint64_t num_samples, XPtrTorchbool replacement, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_multinomial_tensor_intt_bool_generator(self.get(), num_samples.get(), replacement.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lgamma__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_lgamma__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lgamma_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_lgamma_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_digamma_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_digamma_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_polygamma__self_Tensor_n_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_Tensor_polygamma__tensor_intt(self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfinv_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erfinv_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_erfinv__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_erfinv__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_i0_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_i0_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_i0__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_i0__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sign_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sign_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_sign__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_sign__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_signbit_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_signbit_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_dist_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar p) {
  auto r_out = lantern_Tensor_dist_tensor_tensor_scalar(self.get(), other.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan2__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_atan2__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_atan2_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_atan2_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Scalar (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchScalar weight) {
  auto r_out = lantern_Tensor_lerp_tensor_tensor_scalar(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchTensor weight) {
  auto r_out = lantern_Tensor_lerp_tensor_tensor_tensor(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_histc_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchScalar min, XPtrTorchScalar max) {
  auto r_out = lantern_Tensor_histc_tensor_intt_scalar_scalar(self.get(), bins.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_histogram_self_Tensor_bins_Tensor (XPtrTorchTensor self, XPtrTorchTensor bins, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_Tensor_histogram_tensor_tensor_tensor_bool(self.get(), bins.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_histogram_self_Tensor_bins_int64_t (XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchOptionalDoubleArrayRef range, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_Tensor_histogram_tensor_intt_arrayrefdouble_tensor_bool(self.get(), bins.get(), range.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_fmod_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_fmod__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_fmod_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmod__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_fmod__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hypot_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_hypot_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_hypot__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_hypot__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_igamma_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_igamma_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_igamma__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_igamma__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_igammac_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_igammac_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_igammac__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_igammac__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nextafter_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_nextafter_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nextafter__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_nextafter__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_remainder_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_Tensor_remainder__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_remainder_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_remainder__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_remainder__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_min_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_min_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmin_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_fmin_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_max_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_max_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_fmax_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_fmax_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_maximum_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_maximum_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_max_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_max_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_minimum_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_minimum_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_min_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_min_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_quantile_tensor_double_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_quantile_tensor_tensor_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_nanquantile_tensor_double_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_Tensor_nanquantile_tensor_tensor_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_Tensor_quantile_tensor_double_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_quantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_Tensor_quantile_tensor_tensor_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_Tensor_nanquantile_tensor_double_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_nanquantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_Tensor_nanquantile_tensor_tensor_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_sort_tensor_intt_bool(self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_int64_t_stable_bool (XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_sort_tensor_bool_intt_bool(self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_sort_tensor_dimname_bool(self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_Dimname_stable_bool (XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_sort_tensor_bool_dimname_bool(self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_msort_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_msort_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argsort_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_argsort_tensor_intt_bool(self.get(), dim.get(), descending.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_argsort_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_Tensor_argsort_tensor_dimname_bool(self.get(), dim.get(), descending.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_topk_self_Tensor_k_int64_t (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool largest, XPtrTorchbool sorted) {
  auto r_out = lantern_Tensor_topk_tensor_intt_intt_bool_bool(self.get(), k.get(), dim.get(), largest.get(), sorted.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_all_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_all_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_any_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_any_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (XPtrTorchTensor self, XPtrTorchScalar p, XPtrTorchindex_int64_t dim, XPtrTorchScalar maxnorm) {
  auto r_out = lantern_Tensor_renorm_tensor_scalar_intt_scalar(self.get(), p.get(), dim.get(), maxnorm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_renorm__self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (XPtrTorchTensor self, XPtrTorchScalar p, XPtrTorchindex_int64_t dim, XPtrTorchScalar maxnorm) {
  auto r_out = lantern_Tensor_renorm__tensor_scalar_intt_scalar(self.get(), p.get(), dim.get(), maxnorm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_unfold_self_Tensor_dimension_int64_t_size_int64_t_step_int64_t (XPtrTorchTensor self, XPtrTorchint64_t dimension, XPtrTorchint64_t size, XPtrTorchint64_t step) {
  auto r_out = lantern_Tensor_unfold_tensor_intt_intt_intt(self.get(), dimension.get(), size.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_method_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow_self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_Tensor_pow_tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow_self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_Tensor_pow_tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow__self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_Tensor_pow__tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_pow__self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_Tensor_pow__tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_float_power_self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_Tensor_float_power_tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_float_power_self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_Tensor_float_power_tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_float_power__self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_Tensor_float_power__tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_float_power__self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_Tensor_float_power__tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_normal__self_Tensor (XPtrTorchTensor self, XPtrTorchdouble mean, XPtrTorchdouble std, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_Tensor_normal__tensor_double_double_generator(self.get(), mean.get(), std.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_alias_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_alias_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isfinite_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isfinite_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isinf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isinf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_method_record_stream_self_Tensor_s_Stream (XPtrTorchTensor self, XPtrTorch s) {
  lantern_Tensor_record_stream_tensor_stream(self.get(), s.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isposinf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isposinf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_isneginf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_isneginf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_det_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_Tensor_det_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_inner_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_Tensor_inner_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_outer_self_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_Tensor_outer_tensor_tensor(self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_method_ger_self_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_Tensor_ger_tensor_tensor(self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Byte_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_byte_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Char_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_char_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Double_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_double_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Float_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_float_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Int_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_int_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Long_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_long_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Short_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_short_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cast_Half_self_Tensor (XPtrTorchTensor self, XPtrTorchbool non_blocking) {
  auto r_out = lantern__cast_half_tensor_bool(self.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__make_dual_primal_Tensor_tangent_Tensor_level_int64_t (XPtrTorchTensor primal, XPtrTorchTensor tangent, XPtrTorchint64_t level) {
  auto r_out = lantern__make_dual_tensor_tensor_intt(primal.get(), tangent.get(), level.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unpack_dual_dual_Tensor_level_int64_t (XPtrTorchTensor dual, XPtrTorchint64_t level) {
  auto r_out = lantern__unpack_dual_tensor_intt(dual.get(), level.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_align_tensors_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_align_tensors_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__assert_async_self_Tensor (XPtrTorchTensor self) {
  lantern__assert_async_tensor(self.get());
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace__use_cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t (XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchIntArrayRef input_lengths, XPtrTorchIntArrayRef target_lengths, XPtrTorchint64_t blank) {
  auto r_out = lantern__use_cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt(log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), blank.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t_deterministic_bool_zero_infinity_bool (XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchIntArrayRef input_lengths, XPtrTorchIntArrayRef target_lengths, XPtrTorchint64_t blank, XPtrTorchbool deterministic, XPtrTorchbool zero_infinity) {
  auto r_out = lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool(log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), blank.get(), deterministic.get(), zero_infinity.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cudnn_rnn_flatten_weight_weight_arr_TensorList_weight_stride0_int64_t_input_size_int64_t_mode_int64_t_hidden_size_int64_t_proj_size_int64_t_num_layers_int64_t_batch_first_bool_bidirectional_bool (XPtrTorchTensorList weight_arr, XPtrTorchint64_t weight_stride0, XPtrTorchint64_t input_size, XPtrTorchint64_t mode, XPtrTorchint64_t hidden_size, XPtrTorchint64_t proj_size, XPtrTorchint64_t num_layers, XPtrTorchbool batch_first, XPtrTorchbool bidirectional) {
  auto r_out = lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_intt_bool_bool(weight_arr.get(), weight_stride0.get(), input_size.get(), mode.get(), hidden_size.get(), proj_size.get(), num_layers.get(), batch_first.get(), bidirectional.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_proj_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (XPtrTorchTensor input, XPtrTorchTensorList weight, XPtrTorchint64_t weight_stride0, XPtrTorchOptionalTensor weight_buf, XPtrTorchTensor hx, XPtrTorchOptionalTensor cx, XPtrTorchint64_t mode, XPtrTorchint64_t hidden_size, XPtrTorchint64_t proj_size, XPtrTorchint64_t num_layers, XPtrTorchbool batch_first, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchIntArrayRef batch_sizes, XPtrTorchOptionalTensor dropout_state) {
  auto r_out = lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(input.get(), weight.get(), weight_stride0.get(), weight_buf.get(), hx.get(), cx.get(), mode.get(), hidden_size.get(), proj_size.get(), num_layers.get(), batch_first.get(), dropout.get(), train.get(), bidirectional.get(), batch_sizes.get(), dropout_state.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_proj_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (XPtrTorchTensor input, XPtrTorchTensorList weight, XPtrTorchint64_t weight_stride0, XPtrTorchTensor weight_buf, XPtrTorchTensor hx, XPtrTorchOptionalTensor cx, XPtrTorchTensor output, XPtrTorchOptionalTensor grad_output, XPtrTorchOptionalTensor grad_hy, XPtrTorchOptionalTensor grad_cy, XPtrTorchint64_t mode, XPtrTorchint64_t hidden_size, XPtrTorchint64_t proj_size, XPtrTorchint64_t num_layers, XPtrTorchbool batch_first, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchIntArrayRef batch_sizes, XPtrTorchOptionalTensor dropout_state, XPtrTorchTensor reserve, std::vector<bool> output_mask) {
  auto r_out = lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(input.get(), weight.get(), weight_stride0.get(), weight_buf.get(), hx.get(), cx.get(), output.get(), grad_output.get(), grad_hy.get(), grad_cy.get(), mode.get(), hidden_size.get(), proj_size.get(), num_layers.get(), batch_first.get(), dropout.get(), train.get(), bidirectional.get(), batch_sizes.get(), dropout_state.get(), reserve.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensorList(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cudnn_init_dropout_state_dropout_double_train_bool_dropout_seed_int64_t_options_TensorOptions (XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchint64_t dropout_seed, XPtrTorchTensorOptions options) {
  auto r_out = lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions(dropout.get(), train.get(), dropout_seed.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace__debug_has_internal_overlap_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__debug_has_internal_overlap_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fused_dropout_self_Tensor_p_double (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern__fused_dropout_tensor_double_generator(self.get(), p.get(), generator.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__masked_scale_self_Tensor_mask_Tensor_scale_double (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchdouble scale) {
  auto r_out = lantern__masked_scale_tensor_tensor_double(self.get(), mask.get(), scale.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__sobol_engine_draw_quasi_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t_dtype_ScalarType (XPtrTorchTensor quasi, XPtrTorchint64_t n, XPtrTorchTensor sobolstate, XPtrTorchint64_t dimension, XPtrTorchint64_t num_generated, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype(quasi.get(), n.get(), sobolstate.get(), dimension.get(), num_generated.get(), dtype.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_ff__self_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n, XPtrTorchTensor sobolstate, XPtrTorchint64_t dimension, XPtrTorchint64_t num_generated) {
  auto r_out = lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(self.get(), n.get(), sobolstate.get(), dimension.get(), num_generated.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_scramble__self_Tensor_ltm_Tensor_dimension_int64_t (XPtrTorchTensor self, XPtrTorchTensor ltm, XPtrTorchint64_t dimension) {
  auto r_out = lantern__sobol_engine_scramble__tensor_tensor_intt(self.get(), ltm.get(), dimension.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sobol_engine_initialize_state__self_Tensor_dimension_int64_t (XPtrTorchTensor self, XPtrTorchint64_t dimension) {
  auto r_out = lantern__sobol_engine_initialize_state__tensor_intt(self.get(), dimension.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__reshape_from_tensor_self_Tensor_shape_Tensor (XPtrTorchTensor self, XPtrTorchTensor shape) {
  auto r_out = lantern__reshape_from_tensor_tensor_tensor(self.get(), shape.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__shape_as_tensor_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__shape_as_tensor_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dropout_input_Tensor_p_double_train_bool (XPtrTorchTensor input, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_dropout_tensor_double_bool(input.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dropout__self_Tensor_p_double_train_bool (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_dropout__tensor_double_bool(self.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_dropout_input_Tensor_p_double_train_bool (XPtrTorchTensor input, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_feature_dropout_tensor_double_bool(input.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_dropout__self_Tensor_p_double_train_bool (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_feature_dropout__tensor_double_bool(self.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alpha_dropout_input_Tensor_p_double_train_bool (XPtrTorchTensor input, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_alpha_dropout_tensor_double_bool(input.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alpha_dropout__self_Tensor_p_double_train_bool (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_alpha_dropout__tensor_double_bool(self.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_alpha_dropout_input_Tensor_p_double_train_bool (XPtrTorchTensor input, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_feature_alpha_dropout_tensor_double_bool(input.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_feature_alpha_dropout__self_Tensor_p_double_train_bool (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchbool train) {
  auto r_out = lantern_feature_alpha_dropout__tensor_double_bool(self.get(), p.get(), train.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_abs_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_abs__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_abs_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_abs_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_absolute_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_absolute_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_absolute_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_absolute_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_angle_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_angle_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_angle_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_angle_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_view_as_real_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_view_as_real_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_view_as_complex_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_view_as_complex_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sgn_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sgn_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sgn_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sgn_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_real_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_real_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_imag_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_imag_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__conj_physical_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__conj_physical_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_physical_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_conj_physical_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_physical_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_conj_physical_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conj_physical__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_conj_physical__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_resolve_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_resolve_conj_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_resolve_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_resolve_neg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__neg_view_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__neg_view_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_acos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_acos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acos_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_acos_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arccos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arccos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccos_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arccos_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool1d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad) {
  auto r_out = lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool1d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_avg_pool1d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool1d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_max_pool1d_tensor_intarrayref(self.get(), output_size.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_add_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_add_out_tensor_tensor_tensor_scalar(out.get(), self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern__add_relu_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern__add_relu__tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern__add_relu_out_tensor_tensor_tensor_scalar(out.get(), self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern__add_relu_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_relu__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern__add_relu__tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_add_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_add_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv_self_Tensor_mat_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat, XPtrTorchTensor vec, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addmv_tensor_tensor_tensor_scalar_scalar(self.get(), mat.get(), vec.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv__self_Tensor_mat_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat, XPtrTorchTensor vec, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addmv__tensor_tensor_tensor_scalar_scalar(self.get(), mat.get(), vec.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmv_out_out_Tensor_self_Tensor_mat_Tensor_vec_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mat, XPtrTorchTensor vec, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), mat.get(), vec.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addr_self_Tensor_vec1_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec1, XPtrTorchTensor vec2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addr_tensor_tensor_tensor_scalar_scalar(self.get(), vec1.get(), vec2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addr_out_out_Tensor_self_Tensor_vec1_Tensor_vec2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor vec1, XPtrTorchTensor vec2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), vec1.get(), vec2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_affine_grid_generator_theta_Tensor_size_IntArrayRef_align_corners_bool (XPtrTorchTensor theta, XPtrTorchIntArrayRef size, XPtrTorchbool align_corners) {
  auto r_out = lantern_affine_grid_generator_tensor_intarrayref_bool(theta.get(), size.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_affine_grid_generator_backward_grad_Tensor_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad, XPtrTorchIntArrayRef size, XPtrTorchbool align_corners) {
  auto r_out = lantern_affine_grid_generator_backward_tensor_intarrayref_bool(grad.get(), size.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_all_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_all_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_all_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_all_out_tensor_tensor_dimname_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_allclose_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchdouble rtol, XPtrTorchdouble atol, XPtrTorchbool equal_nan) {
  auto r_out = lantern_allclose_tensor_tensor_double_double_bool(self.get(), other.get(), rtol.get(), atol.get(), equal_nan.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_any_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_any_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_any_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_any_out_tensor_tensor_dimname_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_end_Scalar (XPtrTorchScalar end, XPtrTorchTensorOptions options) {
  auto r_out = lantern_arange_scalar_tensoroptions(end.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_start_Scalar_end_Scalar (XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchTensorOptions options) {
  auto r_out = lantern_arange_scalar_scalar_tensoroptions(start.get(), end.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_start_Scalar_end_Scalar_step_Scalar (XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchScalar step, XPtrTorchTensorOptions options) {
  auto r_out = lantern_arange_scalar_scalar_scalar_tensoroptions(start.get(), end.get(), step.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_out_out_Tensor_end_Scalar (XPtrTorchTensor out, XPtrTorchScalar end) {
  auto r_out = lantern_arange_out_tensor_scalar(out.get(), end.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arange_out_out_Tensor_start_Scalar_end_Scalar (XPtrTorchTensor out, XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchScalar step) {
  auto r_out = lantern_arange_out_tensor_scalar_scalar_scalar(out.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__dim_arange_like_Tensor_dim_int64_t (XPtrTorchTensor like, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__dim_arange_tensor_intt(like.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmax_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_argmax_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmax_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_argmax_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmin_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_argmin_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argmin_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_argmin_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_acosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_acosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_acosh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_acosh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arccosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arccosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arccosh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arccosh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_asinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_asinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asinh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_asinh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arcsinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arcsinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsinh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arcsinh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atanh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_atanh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arctanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arctanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctanh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arctanh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchoptional_int64_t storage_offset) {
  auto r_out = lantern_as_strided_tensor_intarrayref_intarrayref_intt(self.get(), size.get(), stride.get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchoptional_int64_t storage_offset) {
  auto r_out = lantern_as_strided__tensor_intarrayref_intarrayref_intt(self.get(), size.get(), stride.get(), storage_offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_asin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_asin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_asin_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_asin_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arcsin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arcsin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arcsin_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arcsin_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_atan_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arctan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_arctan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_arctan_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_arctan_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_1d_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atleast_1d_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_atleast_1d_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_atleast_1d_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_2d_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atleast_2d_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_atleast_2d_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_atleast_2d_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atleast_3d_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_atleast_3d_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_atleast_3d_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_atleast_3d_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_baddbmm_tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__baddbmm_mkl__self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_baddbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bartlett_window_window_length_int64_t (XPtrTorchint64_t window_length, XPtrTorchTensorOptions options) {
  auto r_out = lantern_bartlett_window_intt_tensoroptions(window_length.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bartlett_window_window_length_int64_t_periodic_bool (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchTensorOptions options) {
  auto r_out = lantern_bartlett_window_intt_bool_tensoroptions(window_length.get(), periodic.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble momentum, XPtrTorchdouble eps, XPtrTorchbool cudnn_enabled) {
  auto r_out = lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), momentum.get(), eps.get(), cudnn_enabled.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_var_Tensor_eps_double_output_scale_double_output_zero_point_int64_t (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchTensor mean, XPtrTorchTensor var, XPtrTorchdouble eps, XPtrTorchdouble output_scale, XPtrTorchint64_t output_zero_point) {
  auto r_out = lantern_quantized_batch_norm_tensor_tensor_tensor_tensor_tensor_double_double_intt(input.get(), weight.get(), bias.get(), mean.get(), var.get(), eps.get(), output_scale.get(), output_zero_point.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble momentum, XPtrTorchdouble eps, XPtrTorchbool cudnn_enabled) {
  auto r_out = lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), momentum.get(), eps.get(), cudnn_enabled.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchint64_t(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_backward_impl_index_int64_t_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_transform_Tensor_train_bool_eps_double_output_mask_stdarraybool3_reservedSpace_Tensor (XPtrTorchint64_t impl_index, XPtrTorchTensor input, XPtrTorchTensor grad_output, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchOptionalTensor save_mean, XPtrTorchOptionalTensor save_var_transform, XPtrTorchbool train, XPtrTorchdouble eps, std::vector<bool> output_mask, XPtrTorchTensor reservedSpace) {
  auto r_out = lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool_tensor(impl_index.get(), input.get(), grad_output.get(), weight.get(), running_mean.get(), running_var.get(), save_mean.get(), save_var_transform.get(), train.get(), eps.get(), reinterpret_cast<void*>(&output_mask), reservedSpace.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_bernoulli_tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_bernoulli_out_tensor_tensor_generator(out.get(), self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bernoulli_self_Tensor_p_double (XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_bernoulli_tensor_double_generator(self.get(), p.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bilinear_input1_Tensor_input2_Tensor_weight_Tensor_bias_Tensor (XPtrTorchTensor input1, XPtrTorchTensor input2, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias) {
  auto r_out = lantern_bilinear_tensor_tensor_tensor_tensor(input1.get(), input2.get(), weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_tensor_tensor_tensor_intt(self.get(), target.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt(out.get(), self.get(), target.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_backward_grad_output_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt(grad_output.get(), self.get(), target.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt(grad_input.get(), grad_output.get(), self.get(), target.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_with_logits_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor pos_weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(self.get(), target.get(), weight.get(), pos_weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binary_cross_entropy_with_logits_backward_grad_output_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor pos_weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(grad_output.get(), self.get(), target.get(), weight.get(), pos_weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bincount_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor weights, XPtrTorchint64_t minlength) {
  auto r_out = lantern_bincount_tensor_tensor_intt(self.get(), weights.get(), minlength.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_not_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_bitwise_not_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_not_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_bitwise_not_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copysign_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_copysign_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copysign_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_copysign_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copysign_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_copysign_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copysign_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_copysign_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_not_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_logical_not_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_not_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_logical_not_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_xor_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_xor_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_xor_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_xor_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_and_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_and_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_and_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_and_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_or_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_or_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logical_or_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logical_or_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_blackman_window_window_length_int64_t (XPtrTorchint64_t window_length, XPtrTorchTensorOptions options) {
  auto r_out = lantern_blackman_window_intt_tensoroptions(window_length.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_blackman_window_window_length_int64_t_periodic_bool (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchTensorOptions options) {
  auto r_out = lantern_blackman_window_intt_bool_tensoroptions(window_length.get(), periodic.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bmm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_bmm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bmm_out_out_Tensor_self_Tensor_mat2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_bmm_out_tensor_tensor_tensor(out.get(), self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_broadcast_tensors_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_broadcast_tensors_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_broadcast_to_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_broadcast_to_tensor_intarrayref(self.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_tensors_TensorList_dim_int64_t (XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cat_tensorlist_intt(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cat_out_tensor_tensorlist_intt(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_tensors_TensorList_dim_Dimname (XPtrTorchTensorList tensors, XPtrTorchDimname dim) {
  auto r_out = lantern_cat_tensorlist_dimname(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchDimname dim) {
  auto r_out = lantern_cat_out_tensor_tensorlist_dimname(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_concat_tensors_TensorList_dim_int64_t (XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_concat_tensorlist_intt(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_concat_out_out_Tensor_tensors_TensorList_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_concat_out_tensor_tensorlist_intt(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_concat_tensors_TensorList_dim_Dimname (XPtrTorchTensorList tensors, XPtrTorchDimname dim) {
  auto r_out = lantern_concat_tensorlist_dimname(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_concat_out_out_Tensor_tensors_TensorList_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchDimname dim) {
  auto r_out = lantern_concat_out_tensor_tensorlist_dimname(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_block_diag_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_block_diag_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_ceil_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_ceil__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ceil_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_ceil_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_chain_matmul_matrices_TensorList (XPtrTorchTensorList matrices) {
  auto r_out = lantern_chain_matmul_tensorlist(matrices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_chain_matmul_out_out_Tensor_matrices_TensorList (XPtrTorchTensor out, XPtrTorchTensorList matrices) {
  auto r_out = lantern_chain_matmul_out_tensor_tensorlist(out.get(), matrices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unsafe_chunk_self_Tensor_chunks_int64_t (XPtrTorchTensor self, XPtrTorchint64_t chunks, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_unsafe_chunk_tensor_intt_intt(self.get(), chunks.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_chunk_self_Tensor_chunks_int64_t (XPtrTorchTensor self, XPtrTorchint64_t chunks, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_chunk_tensor_intt_intt(self.get(), chunks.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_tensor_split_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_tensor_split_tensor_intt_intt(self.get(), sections.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_tensor_split_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_tensor_split_tensor_intarrayref_intt(self.get(), indices.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_tensor_split_self_Tensor_tensor_indices_or_sections_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor_indices_or_sections, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_tensor_split_tensor_tensor_intt(self.get(), tensor_indices_or_sections.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clamp_tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clamp_tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp__self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clamp__tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp__self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clamp__tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_out_out_Tensor_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clamp_out_tensor_tensor_scalar_scalar(out.get(), self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_out_out_Tensor_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clamp_out_tensor_tensor_tensor_tensor(out.get(), self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_self_Tensor_max_Scalar (XPtrTorchTensor self, XPtrTorchScalar max) {
  auto r_out = lantern_clamp_max_tensor_scalar(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_self_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchTensor max) {
  auto r_out = lantern_clamp_max_tensor_tensor(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max__self_Tensor_max_Scalar (XPtrTorchTensor self, XPtrTorchScalar max) {
  auto r_out = lantern_clamp_max__tensor_scalar(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max__self_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchTensor max) {
  auto r_out = lantern_clamp_max__tensor_tensor(self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_out_out_Tensor_self_Tensor_max_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar max) {
  auto r_out = lantern_clamp_max_out_tensor_tensor_scalar(out.get(), self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_max_out_out_Tensor_self_Tensor_max_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor max) {
  auto r_out = lantern_clamp_max_out_tensor_tensor_tensor(out.get(), self.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_self_Tensor_min_Scalar (XPtrTorchTensor self, XPtrTorchScalar min) {
  auto r_out = lantern_clamp_min_tensor_scalar(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_self_Tensor_min_Tensor (XPtrTorchTensor self, XPtrTorchTensor min) {
  auto r_out = lantern_clamp_min_tensor_tensor(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min__self_Tensor_min_Scalar (XPtrTorchTensor self, XPtrTorchScalar min) {
  auto r_out = lantern_clamp_min__tensor_scalar(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min__self_Tensor_min_Tensor (XPtrTorchTensor self, XPtrTorchTensor min) {
  auto r_out = lantern_clamp_min__tensor_tensor(self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_out_out_Tensor_self_Tensor_min_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar min) {
  auto r_out = lantern_clamp_min_out_tensor_tensor_scalar(out.get(), self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clamp_min_out_out_Tensor_self_Tensor_min_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor min) {
  auto r_out = lantern_clamp_min_out_tensor_tensor_tensor(out.get(), self.get(), min.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clip_tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clip_tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip__self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clip__tensor_scalar_scalar(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip__self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clip__tensor_tensor_tensor(self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_out_out_Tensor_self_Tensor_min_Scalar_max_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar min, XPtrTorchoptional_scalar max) {
  auto r_out = lantern_clip_out_tensor_tensor_scalar_scalar(out.get(), self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clip_out_out_Tensor_self_Tensor_min_Tensor_max_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalTensor min, XPtrTorchOptionalTensor max) {
  auto r_out = lantern_clip_out_tensor_tensor_tensor_tensor(out.get(), self.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_cudnn_is_acceptable_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_cudnn_is_acceptable_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_complex_real_Tensor_imag_Tensor (XPtrTorchTensor real, XPtrTorchTensor imag) {
  auto r_out = lantern_complex_tensor_tensor(real.get(), imag.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_complex_out_out_Tensor_real_Tensor_imag_Tensor (XPtrTorchTensor out, XPtrTorchTensor real, XPtrTorchTensor imag) {
  auto r_out = lantern_complex_out_tensor_tensor_tensor(out.get(), real.get(), imag.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polar_abs_Tensor_angle_Tensor (XPtrTorchTensor abs, XPtrTorchTensor angle) {
  auto r_out = lantern_polar_tensor_tensor(abs.get(), angle.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polar_out_out_Tensor_abs_Tensor_angle_Tensor (XPtrTorchTensor out, XPtrTorchTensor abs, XPtrTorchTensor angle) {
  auto r_out = lantern_polar_out_tensor_tensor_tensor(out.get(), abs.get(), angle.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_constant_pad_nd_self_Tensor_pad_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef pad, XPtrTorchScalar value) {
  auto r_out = lantern_constant_pad_nd_tensor_intarrayref_scalar(self.get(), pad.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups) {
  auto r_out = lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_convolution_overrideable_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups) {
  auto r_out = lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_convolution_backward_overrideable_grad_output_Tensor_input_Tensor_weight_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, std::vector<bool> output_mask) {
  auto r_out = lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool(grad_output.get(), input.get(), weight.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool_allow_tf32_bool (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool cudnn_enabled, XPtrTorchbool allow_tf32) {
  auto r_out = lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get(), benchmark.get(), deterministic.get(), cudnn_enabled.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool cudnn_enabled) {
  auto r_out = lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get(), benchmark.get(), deterministic.get(), cudnn_enabled.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_mode_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_c10string_view_dilation_IntArrayRef_groups_int64_t (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchstring_view padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern__convolution_mode_tensor_tensor_tensor_intarrayref_cstringview_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convolution_nogroup_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding) {
  auto r_out = lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__convolution_double_backward_ggI_Tensor_ggW_Tensor_ggb_Tensor_gO_Tensor_weight_Tensor_self_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool_allow_tf32_bool_output_mask_stdarraybool3 (XPtrTorchOptionalTensor ggI, XPtrTorchOptionalTensor ggW, XPtrTorchOptionalTensor ggb, XPtrTorchTensor gO, XPtrTorchTensor weight, XPtrTorchTensor self, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool transposed, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool cudnn_enabled, XPtrTorchbool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool_stdarraybool(ggI.get(), ggW.get(), ggb.get(), gO.get(), weight.get(), self.get(), stride.get(), padding.get(), dilation.get(), transposed.get(), output_padding.get(), groups.get(), benchmark.get(), deterministic.get(), cudnn_enabled.get(), allow_tf32.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv1d_input_Tensor_weight_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv2d_input_Tensor_weight_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv3d_input_Tensor_weight_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv1d_input_Tensor_weight_Tensor_padding_c10string_view (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchstring_view padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv1d_tensor_tensor_tensor_intarrayref_cstringview_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv2d_input_Tensor_weight_Tensor_padding_c10string_view (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchstring_view padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv2d_tensor_tensor_tensor_intarrayref_cstringview_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv3d_input_Tensor_weight_Tensor_padding_c10string_view (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchstring_view padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_conv3d_tensor_tensor_tensor_intarrayref_cstringview_intarrayref_intt(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_tbc_self_Tensor_weight_Tensor_bias_Tensor (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchTensor bias, XPtrTorchint64_t pad) {
  auto r_out = lantern_conv_tbc_tensor_tensor_tensor_intt(self.get(), weight.get(), bias.get(), pad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_conv_tbc_backward_self_Tensor_input_Tensor_weight_Tensor_bias_Tensor_pad_int64_t (XPtrTorchTensor self, XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchTensor bias, XPtrTorchint64_t pad) {
  auto r_out = lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt(self.get(), input.get(), weight.get(), bias.get(), pad.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose1d_input_Tensor_weight_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), groups.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose2d_input_Tensor_weight_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), groups.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_transpose3d_input_Tensor_weight_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchint64_t groups, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(input.get(), weight.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), groups.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__copy_from_self_Tensor_dst_Tensor (XPtrTorchTensor self, XPtrTorchTensor dst, XPtrTorchbool non_blocking) {
  auto r_out = lantern__copy_from_tensor_tensor_bool(self.get(), dst.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__copy_from_and_resize_self_Tensor_dst_Tensor (XPtrTorchTensor self, XPtrTorchTensor dst) {
  auto r_out = lantern__copy_from_and_resize_tensor_tensor(self.get(), dst.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_cos_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_cos__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cos_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_cos_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_cosh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_cosh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_cosh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosine_embedding_loss_input1_Tensor_input2_Tensor_target_Tensor (XPtrTorchTensor input1, XPtrTorchTensor input2, XPtrTorchTensor target, XPtrTorchdouble margin, XPtrTorchint64_t reduction) {
  auto r_out = lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(input1.get(), input2.get(), target.get(), margin.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_count_nonzero_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim) {
  auto r_out = lantern_count_nonzero_tensor_intarrayref(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_count_nonzero_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_count_nonzero_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cov_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t correction, XPtrTorchOptionalTensor fweights, XPtrTorchOptionalTensor aweights) {
  auto r_out = lantern_cov_tensor_intt_tensor_tensor(self.get(), correction.get(), fweights.get(), aweights.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_corrcoef_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_corrcoef_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_affine_grid_generator_theta_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (XPtrTorchTensor theta, XPtrTorchint64_t False, XPtrTorchint64_t C, XPtrTorchint64_t H, XPtrTorchint64_t W) {
  auto r_out = lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt(theta.get(), False.get(), C.get(), H.get(), W.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_affine_grid_generator_backward_grad_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (XPtrTorchTensor grad, XPtrTorchint64_t False, XPtrTorchint64_t C, XPtrTorchint64_t H, XPtrTorchint64_t W) {
  auto r_out = lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt(grad.get(), False.get(), C.get(), H.get(), W.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble exponential_average_factor, XPtrTorchdouble epsilon) {
  auto r_out = lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), exponential_average_factor.get(), epsilon.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double_reserveSpace_Tensor (XPtrTorchTensor input, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchOptionalTensor save_mean, XPtrTorchOptionalTensor save_var, XPtrTorchdouble epsilon, XPtrTorchTensor reserveSpace) {
  auto r_out = lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_tensor(input.get(), grad_output.get(), weight.get(), running_mean.get(), running_var.get(), save_mean.get(), save_var.get(), epsilon.get(), reserveSpace.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), bias.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_allow_tf32_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(self.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (XPtrTorchIntArrayRef self_size, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(self_size.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool_output_mask_stdarraybool2 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), bias.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_allow_tf32_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(self.get(), weight.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool_output_mask_stdarraybool2 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32, std::vector<bool> output_mask) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_allow_tf32_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, XPtrTorchbool allow_tf32) {
  auto r_out = lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), allow_tf32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_relu_self_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_groups_int64_t (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_cudnn_convolution_relu_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(self.get(), weight.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_convolution_add_relu_self_Tensor_weight_Tensor_z_Tensor_alpha_Scalar_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_groups_int64_t (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchTensor z, XPtrTorchoptional_scalar alpha, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_cudnn_convolution_add_relu_tensor_tensor_tensor_scalar_tensor_intarrayref_intarrayref_intarrayref_intt(self.get(), weight.get(), z.get(), alpha.get(), bias.get(), stride.get(), padding.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cudnn_grid_sampler_self_Tensor_grid_Tensor (XPtrTorchTensor self, XPtrTorchTensor grid) {
  auto r_out = lantern_cudnn_grid_sampler_tensor_tensor(self.get(), grid.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_grid_sampler_backward_self_Tensor_grid_Tensor_grad_output_Tensor (XPtrTorchTensor self, XPtrTorchTensor grid, XPtrTorchTensor grad_output) {
  auto r_out = lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor(self.get(), grid.get(), grad_output.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cummax_tensor_intt(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cummax_out_tensor_tensor_tensor_intt(values.get(), indices.get(), self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_cummax_tensor_dimname(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummax_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_cummax_out_tensor_tensor_tensor_dimname(values.get(), indices.get(), self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
void cpp_torch_namespace__cummax_helper_self_Tensor_values_Tensor_indices_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchindex_int64_t dim) {
  lantern__cummax_helper_tensor_tensor_tensor_intt(self.get(), values.get(), indices.get(), dim.get());
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cummin_tensor_intt(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cummin_out_tensor_tensor_tensor_intt(values.get(), indices.get(), self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_cummin_tensor_dimname(self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cummin_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_cummin_out_tensor_tensor_tensor_dimname(values.get(), indices.get(), self.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
void cpp_torch_namespace__cummin_helper_self_Tensor_values_Tensor_indices_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchindex_int64_t dim) {
  lantern__cummin_helper_tensor_tensor_tensor_intt(self.get(), values.get(), indices.get(), dim.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cummaxmin_backward_grad_Tensor_input_Tensor_indices_Tensor_dim_int64_t (XPtrTorchTensor grad, XPtrTorchTensor input, XPtrTorchIndexTensor indices, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cummaxmin_backward_tensor_tensor_tensor_intt(grad.get(), input.get(), indices.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumprod_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumprod_out_tensor_tensor_intt_scalartype(out.get(), self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumprod_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumprod_out_tensor_tensor_dimname_scalartype(out.get(), self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumprod_backward_grad_Tensor_input_Tensor_dim_int64_t_output_Tensor (XPtrTorchTensor grad, XPtrTorchTensor input, XPtrTorchindex_int64_t dim, XPtrTorchTensor output) {
  auto r_out = lantern_cumprod_backward_tensor_tensor_intt_tensor(grad.get(), input.get(), dim.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumsum_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumsum_out_tensor_tensor_intt_scalartype(out.get(), self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumsum_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_cumsum_out_tensor_tensor_dimname_scalartype(out.get(), self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumulative_trapezoid_y_Tensor_x_Tensor (XPtrTorchTensor y, XPtrTorchTensor x, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cumulative_trapezoid_tensor_tensor_intt(y.get(), x.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cumulative_trapezoid_y_Tensor (XPtrTorchTensor y, XPtrTorchScalar dx, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_cumulative_trapezoid_tensor_scalar_intt(y.get(), dx.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchIntArrayRef input_lengths, XPtrTorchIntArrayRef target_lengths, XPtrTorchint64_t blank, XPtrTorchint64_t reduction, XPtrTorchbool zero_infinity) {
  auto r_out = lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), blank.get(), reduction.get(), zero_infinity.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_Tensor_target_lengths_Tensor (XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchTensor input_lengths, XPtrTorchTensor target_lengths, XPtrTorchint64_t blank, XPtrTorchint64_t reduction, XPtrTorchbool zero_infinity) {
  auto r_out = lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), blank.get(), reduction.get(), zero_infinity.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchIntArrayRef input_lengths, XPtrTorchIntArrayRef target_lengths, XPtrTorchint64_t blank, XPtrTorchbool zero_infinity) {
  auto r_out = lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool(log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), blank.get(), zero_infinity.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__ctc_loss_backward_grad_Tensor_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_neg_log_likelihood_Tensor_log_alpha_Tensor_blank_int64_t (XPtrTorchTensor grad, XPtrTorchTensor log_probs, XPtrTorchTensor targets, XPtrTorchIntArrayRef input_lengths, XPtrTorchIntArrayRef target_lengths, XPtrTorchTensor neg_log_likelihood, XPtrTorchTensor log_alpha, XPtrTorchint64_t blank, XPtrTorchbool zero_infinity) {
  auto r_out = lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool(grad.get(), log_probs.get(), targets.get(), input_lengths.get(), target_lengths.get(), neg_log_likelihood.get(), log_alpha.get(), blank.get(), zero_infinity.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_embed_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t offset, XPtrTorchindex_int64_t dim1, XPtrTorchindex_int64_t dim2) {
  auto r_out = lantern_diag_embed_tensor_intt_intt_intt(self.get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagflat_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t offset) {
  auto r_out = lantern_diagflat_tensor_intt(self.get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_self_Tensor_dim1_int64_t_dim2_int64_t (XPtrTorchTensor self, XPtrTorchint64_t offset, XPtrTorchindex_int64_t dim1, XPtrTorchindex_int64_t dim2) {
  auto r_out = lantern_diagonal_tensor_intt_intt_intt(self.get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_self_Tensor_outdim_Dimname_dim1_Dimname_dim2_Dimname (XPtrTorchTensor self, XPtrTorchDimname outdim, XPtrTorchDimname dim1, XPtrTorchDimname dim2, XPtrTorchint64_t offset) {
  auto r_out = lantern_diagonal_tensor_dimname_dimname_dimname_intt(self.get(), outdim.get(), dim1.get(), dim2.get(), offset.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diagonal_backward_grad_output_Tensor_input_sizes_IntArrayRef_offset_int64_t_dim1_int64_t_dim2_int64_t (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef input_sizes, XPtrTorchint64_t offset, XPtrTorchindex_int64_t dim1, XPtrTorchindex_int64_t dim2) {
  auto r_out = lantern_diagonal_backward_tensor_intarrayref_intt_intt_intt(grad_output.get(), input_sizes.get(), offset.get(), dim1.get(), dim2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diff_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t n, XPtrTorchindex_int64_t dim, XPtrTorchOptionalTensor prepend, XPtrTorchOptionalTensor append) {
  auto r_out = lantern_diff_tensor_intt_intt_tensor_tensor(self.get(), n.get(), dim.get(), prepend.get(), append.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diff_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t n, XPtrTorchindex_int64_t dim, XPtrTorchOptionalTensor prepend, XPtrTorchOptionalTensor append) {
  auto r_out = lantern_diff_out_tensor_tensor_intt_intt_tensor_tensor(out.get(), self.get(), n.get(), dim.get(), prepend.get(), append.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_Scalar_dim_int64_t (XPtrTorchTensor self, XPtrTorchoptional_scalar spacing, XPtrTorchoptional_index_int64_t dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_scalar_intt_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_Scalar_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchScalar spacing, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_scalar_intarrayref_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_intarrayref_intt(self.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_ArrayRefScalar_dim_int64_t (XPtrTorchTensor self, XPtrTorchvector_Scalar spacing, XPtrTorchoptional_index_int64_t dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_arrayrefscalar_intt_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_ArrayRefScalar_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchvector_Scalar spacing, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_arrayrefscalar_intarrayref_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_TensorList_dim_int64_t (XPtrTorchTensor self, XPtrTorchTensorList spacing, XPtrTorchoptional_index_int64_t dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_tensorlist_intt_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_gradient_self_Tensor_spacing_TensorList_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensorList spacing, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t edge_order) {
  auto r_out = lantern_gradient_tensor_tensorlist_intarrayref_intt(self.get(), spacing.get(), dim.get(), edge_order.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_div_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_div_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_div_tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_out_out_Tensor_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_div_out_tensor_tensor_tensor_cstringview(out.get(), self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_div_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_div_self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_div_tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_divide_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_divide_tensor_tensor_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_out_out_Tensor_self_Tensor_other_Tensor_rounding_mode_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_divide_out_tensor_tensor_tensor_cstringview(out.get(), self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_divide_self_Tensor_other_Scalar_rounding_mode_c10string_view (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchoptional_string_view rounding_mode) {
  auto r_out = lantern_divide_tensor_scalar_cstringview(self.get(), other.get(), rounding_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_true_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_true_divide_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_true_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_true_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dot_self_Tensor_tensor_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor) {
  auto r_out = lantern_dot_tensor_tensor(self.get(), tensor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dot_out_out_Tensor_self_Tensor_tensor_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor tensor) {
  auto r_out = lantern_dot_out_tensor_tensor_tensor(out.get(), self.get(), tensor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vdot_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_vdot_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vdot_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_vdot_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_einsum_equation_c10string_view_tensors_TensorList (XPtrTorchstring_view equation, XPtrTorchTensorList tensors) {
  auto r_out = lantern_einsum_cstringview_tensorlist(equation.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_weight_Tensor_indices_Tensor (XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchint64_t padding_idx, XPtrTorchbool scale_grad_by_freq, XPtrTorchbool sparse) {
  auto r_out = lantern_embedding_tensor_tensor_intt_bool_bool(weight.get(), indices.get(), padding_idx.get(), scale_grad_by_freq.get(), sparse.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool_sparse_bool (XPtrTorchTensor grad, XPtrTorchIndexTensor indices, XPtrTorchint64_t num_weights, XPtrTorchint64_t padding_idx, XPtrTorchbool scale_grad_by_freq, XPtrTorchbool sparse) {
  auto r_out = lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(grad.get(), indices.get(), num_weights.get(), padding_idx.get(), scale_grad_by_freq.get(), sparse.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_dense_backward_grad_output_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (XPtrTorchTensor grad_output, XPtrTorchIndexTensor indices, XPtrTorchint64_t num_weights, XPtrTorchint64_t padding_idx, XPtrTorchbool scale_grad_by_freq) {
  auto r_out = lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool(grad_output.get(), indices.get(), num_weights.get(), padding_idx.get(), scale_grad_by_freq.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_renorm__self_Tensor_indices_Tensor_max_norm_double_norm_type_double (XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchdouble max_norm, XPtrTorchdouble norm_type) {
  auto r_out = lantern_embedding_renorm__tensor_tensor_double_double(self.get(), indices.get(), max_norm.get(), norm_type.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_embedding_sparse_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (XPtrTorchTensor grad, XPtrTorchIndexTensor indices, XPtrTorchint64_t num_weights, XPtrTorchint64_t padding_idx, XPtrTorchbool scale_grad_by_freq) {
  auto r_out = lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(grad.get(), indices.get(), num_weights.get(), padding_idx.get(), scale_grad_by_freq.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__embedding_bag_forward_only_weight_Tensor_indices_Tensor_offsets_Tensor (XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchbool sparse, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchbool include_last_offset, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_forward_only_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(weight.get(), indices.get(), offsets.get(), scale_grad_by_freq.get(), mode.get(), sparse.get(), per_sample_weights.get(), include_last_offset.get(), padding_idx.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__rowwise_prune_weight_Tensor_mask_Tensor_compressed_indices_dtype_ScalarType (XPtrTorchTensor weight, XPtrTorchTensor mask, XPtrTorchDtype compressed_indices_dtype) {
  auto r_out = lantern__rowwise_prune_tensor_tensor_scalartype(weight.get(), mask.get(), compressed_indices_dtype.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_row_stack_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_row_stack_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_row_stack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_row_stack_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor_scale_grad_by_freq_bool_mode_int64_t_sparse_bool_per_sample_weights_Tensor_include_last_offset_bool (XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchbool sparse, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchbool include_last_offset) {
  auto r_out = lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(weight.get(), indices.get(), offsets.get(), scale_grad_by_freq.get(), mode.get(), sparse.get(), per_sample_weights.get(), include_last_offset.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor_scale_grad_by_freq_bool_mode_int64_t_sparse_bool_per_sample_weights_Tensor_include_last_offset_bool_padding_idx_int64_t (XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchbool sparse, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchbool include_last_offset, XPtrTorchoptional_int64_t padding_idx) {
  auto r_out = lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(weight.get(), indices.get(), offsets.get(), scale_grad_by_freq.get(), mode.get(), sparse.get(), per_sample_weights.get(), include_last_offset.get(), padding_idx.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor (XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchbool sparse, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchbool include_last_offset, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(weight.get(), indices.get(), offsets.get(), scale_grad_by_freq.get(), mode.get(), sparse.get(), per_sample_weights.get(), include_last_offset.get(), padding_idx.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_sparse_bool_per_sample_weights_Tensor (XPtrTorchTensor grad, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchTensor offset2bag, XPtrTorchTensor bag_size, XPtrTorchTensor maximum_indices, XPtrTorchint64_t num_weights, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchbool sparse, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor_intt(grad.get(), indices.get(), offsets.get(), offset2bag.get(), bag_size.get(), maximum_indices.get(), num_weights.get(), scale_grad_by_freq.get(), mode.get(), sparse.get(), per_sample_weights.get(), padding_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_sparse_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (XPtrTorchTensor grad, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchTensor offset2bag, XPtrTorchTensor bag_size, XPtrTorchint64_t num_weights, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor_intt(grad.get(), indices.get(), offsets.get(), offset2bag.get(), bag_size.get(), num_weights.get(), scale_grad_by_freq.get(), mode.get(), per_sample_weights.get(), padding_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_dense_backward_grad_Tensor_indices_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (XPtrTorchTensor grad, XPtrTorchIndexTensor indices, XPtrTorchTensor offset2bag, XPtrTorchTensor bag_size, XPtrTorchTensor maximum_indices, XPtrTorchint64_t num_weights, XPtrTorchbool scale_grad_by_freq, XPtrTorchint64_t mode, XPtrTorchOptionalTensor per_sample_weights, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor_intt(grad.get(), indices.get(), offset2bag.get(), bag_size.get(), maximum_indices.get(), num_weights.get(), scale_grad_by_freq.get(), mode.get(), per_sample_weights.get(), padding_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__embedding_bag_per_sample_weights_backward_grad_Tensor_weight_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_mode_int64_t (XPtrTorchTensor grad, XPtrTorchTensor weight, XPtrTorchIndexTensor indices, XPtrTorchTensor offsets, XPtrTorchTensor offset2bag, XPtrTorchint64_t mode, XPtrTorchint64_t padding_idx) {
  auto r_out = lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt_intt(grad.get(), weight.get(), indices.get(), offsets.get(), offset2bag.get(), mode.get(), padding_idx.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_size_IntArrayRef_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(size.get(), names.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_empty_intarrayref_tensoroptions_memoryformat(size.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__empty_affine_quantized_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options, XPtrTorchdouble scale, XPtrTorchint64_t zero_point, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat(size.get(), options.get(), scale.get(), zero_point.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__empty_per_channel_affine_quantized_size_IntArrayRef_scales_Tensor_zero_points_Tensor_axis_int64_t (XPtrTorchIntArrayRef size, XPtrTorchTensor scales, XPtrTorchTensor zero_points, XPtrTorchint64_t axis, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat(size.get(), scales.get(), zero_points.get(), axis.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_quantized_size_IntArrayRef_qtensor_Tensor (XPtrTorchIntArrayRef size, XPtrTorchTensor qtensor, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_empty_quantized_intarrayref_tensor_tensoroptions_memoryformat(size.get(), qtensor.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_out_out_Tensor_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIntArrayRef size, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_empty_out_tensor_intarrayref_memoryformat(out.get(), size.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_like_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_empty_like_tensor_tensoroptions_memoryformat(self.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_empty_strided_size_IntArrayRef_stride_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride, XPtrTorchTensorOptions options) {
  auto r_out = lantern_empty_strided_intarrayref_intarrayref_tensoroptions(size.get(), stride.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_erf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_erf__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erf_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_erf_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_erfc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_erfc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_erfc_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_exp_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_exp__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_exp_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_exp2_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_exp2__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_exp2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_exp2_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_expm1_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_expm1__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_expm1_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_expm1_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_n_int64_t (XPtrTorchint64_t n, XPtrTorchTensorOptions options) {
  auto r_out = lantern_eye_intt_tensoroptions(n.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_n_int64_t_m_int64_t (XPtrTorchint64_t n, XPtrTorchint64_t m, XPtrTorchTensorOptions options) {
  auto r_out = lantern_eye_intt_intt_tensoroptions(n.get(), m.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_out_out_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchint64_t n) {
  auto r_out = lantern_eye_out_tensor_intt(out.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eye_out_out_Tensor_n_int64_t_m_int64_t (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchint64_t m) {
  auto r_out = lantern_eye_out_tensor_intt_intt(out.get(), n.get(), m.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t start_dim, XPtrTorchindex_int64_t end_dim) {
  auto r_out = lantern_flatten_tensor_intt_intt(self.get(), start_dim.get(), end_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchindex_int64_t start_dim, XPtrTorchindex_int64_t end_dim, XPtrTorchDimname out_dim) {
  auto r_out = lantern_flatten_tensor_intt_intt_dimname(self.get(), start_dim.get(), end_dim.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname start_dim, XPtrTorchDimname end_dim, XPtrTorchDimname out_dim) {
  auto r_out = lantern_flatten_tensor_dimname_dimname_dimname(self.get(), start_dim.get(), end_dim.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimnameList dims, XPtrTorchDimname out_dim) {
  auto r_out = lantern_flatten_tensor_dimnamelist_dimname(self.get(), dims.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fill__self_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar value) {
  auto r_out = lantern_fill__tensor_scalar(self.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fill__self_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchTensor value) {
  auto r_out = lantern_fill__tensor_tensor(self.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_floor_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_floor__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_floor_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_floor_divide_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_floor_divide_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_floor_divide_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_floor_divide_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_frac_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_frac__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frac_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_frac_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchScalar fill_value, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(size.get(), fill_value.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar (XPtrTorchIntArrayRef size, XPtrTorchScalar fill_value, XPtrTorchTensorOptions options) {
  auto r_out = lantern_full_intarrayref_scalar_tensoroptions(size.get(), fill_value.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_out_out_Tensor_size_IntArrayRef_fill_value_Scalar (XPtrTorchTensor out, XPtrTorchIntArrayRef size, XPtrTorchScalar fill_value) {
  auto r_out = lantern_full_out_tensor_intarrayref_scalar(out.get(), size.get(), fill_value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_full_like_self_Tensor_fill_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar fill_value, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_full_like_tensor_scalar_tensoroptions_memoryformat(self.get(), fill_value.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_from_file_filename_c10string_view (XPtrTorchstring_view filename, XPtrTorchoptional_bool shared, XPtrTorchoptional_int64_t size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_from_file_cstringview_bool_intt_tensoroptions(filename.get(), shared.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_gcd_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_gcd_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gcd__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_gcd__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_lcm_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_lcm_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lcm__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_lcm__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern_grid_sampler_tensor_tensor_intt_intt_bool(input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_2d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool(input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_2d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool(grad_output.get(), input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__grid_sampler_2d_cpu_fallback_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern__grid_sampler_2d_cpu_fallback_tensor_tensor_intt_intt_bool(input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__grid_sampler_2d_cpu_fallback_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern__grid_sampler_2d_cpu_fallback_backward_tensor_tensor_tensor_intt_intt_bool(grad_output.get(), input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_grid_sampler_3d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool(input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_3d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchTensor input, XPtrTorchTensor grid, XPtrTorchint64_t interpolation_mode, XPtrTorchint64_t padding_mode, XPtrTorchbool align_corners) {
  auto r_out = lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool(grad_output.get(), input.get(), grid.get(), interpolation_mode.get(), padding_mode.get(), align_corners.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hann_window_window_length_int64_t (XPtrTorchint64_t window_length, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hann_window_intt_tensoroptions(window_length.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hann_window_window_length_int64_t_periodic_bool (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hann_window_intt_bool_tensoroptions(window_length.get(), periodic.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t (XPtrTorchint64_t window_length, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hamming_window_intt_tensoroptions(window_length.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hamming_window_intt_bool_tensoroptions(window_length.get(), periodic.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchdouble alpha, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hamming_window_intt_bool_double_tensoroptions(window_length.get(), periodic.get(), alpha.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double_beta_double (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchdouble alpha, XPtrTorchdouble beta, XPtrTorchTensorOptions options) {
  auto r_out = lantern_hamming_window_intt_bool_double_double_tensoroptions(window_length.get(), periodic.get(), alpha.get(), beta.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t (XPtrTorchint64_t window_length, XPtrTorchTensorOptions options) {
  auto r_out = lantern_kaiser_window_intt_tensoroptions(window_length.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t_periodic_bool (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchTensorOptions options) {
  auto r_out = lantern_kaiser_window_intt_bool_tensoroptions(window_length.get(), periodic.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kaiser_window_window_length_int64_t_periodic_bool_beta_double (XPtrTorchint64_t window_length, XPtrTorchbool periodic, XPtrTorchdouble beta, XPtrTorchTensorOptions options) {
  auto r_out = lantern_kaiser_window_intt_bool_double_tensoroptions(window_length.get(), periodic.get(), beta.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hinge_embedding_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchdouble margin, XPtrTorchint64_t reduction) {
  auto r_out = lantern_hinge_embedding_loss_tensor_tensor_double_intt(self.get(), target.get(), margin.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_group_norm_input_Tensor_num_groups_int64_t (XPtrTorchTensor input, XPtrTorchint64_t num_groups, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchdouble eps, XPtrTorchbool cudnn_enabled) {
  auto r_out = lantern_group_norm_tensor_intt_tensor_tensor_double_bool(input.get(), num_groups.get(), weight.get(), bias.get(), eps.get(), cudnn_enabled.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_group_norm_input_Tensor_weight_Tensor_bias_Tensor_FALSE_int64_t_C_int64_t_HxW_int64_t_group_int64_t_eps_double (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchint64_t False, XPtrTorchint64_t C, XPtrTorchint64_t HxW, XPtrTorchint64_t group, XPtrTorchdouble eps) {
  auto r_out = lantern_native_group_norm_tensor_tensor_tensor_intt_intt_intt_intt_double(input.get(), weight.get(), bias.get(), False.get(), C.get(), HxW.get(), group.get(), eps.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_group_norm_backward_grad_out_Tensor_input_Tensor_mean_Tensor_rstd_Tensor_weight_Tensor_FALSE_int64_t_C_int64_t_HxW_int64_t_group_int64_t_output_mask_stdarraybool3 (XPtrTorchTensor grad_out, XPtrTorchTensor input, XPtrTorchTensor mean, XPtrTorchTensor rstd, XPtrTorchOptionalTensor weight, XPtrTorchint64_t False, XPtrTorchint64_t C, XPtrTorchint64_t HxW, XPtrTorchint64_t group, std::vector<bool> output_mask) {
  auto r_out = lantern_native_group_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_stdarraybool(grad_out.get(), input.get(), mean.get(), rstd.get(), weight.get(), False.get(), C.get(), HxW.get(), group.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_r2c_self_Tensor_dim_IntArrayRef_normalization_int64_t_onesided_bool (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchbool onesided) {
  auto r_out = lantern__fft_r2c_tensor_intarrayref_intt_bool(self.get(), dim.get(), normalization.get(), onesided.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_r2c_out_out_Tensor_self_Tensor_dim_IntArrayRef_normalization_int64_t_onesided_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchbool onesided) {
  auto r_out = lantern__fft_r2c_out_tensor_tensor_intarrayref_intt_bool(out.get(), self.get(), dim.get(), normalization.get(), onesided.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_c2r_self_Tensor_dim_IntArrayRef_normalization_int64_t_last_dim_size_int64_t (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchint64_t last_dim_size) {
  auto r_out = lantern__fft_c2r_tensor_intarrayref_intt_intt(self.get(), dim.get(), normalization.get(), last_dim_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_c2r_out_out_Tensor_self_Tensor_dim_IntArrayRef_normalization_int64_t_last_dim_size_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchint64_t last_dim_size) {
  auto r_out = lantern__fft_c2r_out_tensor_tensor_intarrayref_intt_intt(out.get(), self.get(), dim.get(), normalization.get(), last_dim_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_c2c_self_Tensor_dim_IntArrayRef_normalization_int64_t_forward_bool (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchbool forward) {
  auto r_out = lantern__fft_c2c_tensor_intarrayref_intt_bool(self.get(), dim.get(), normalization.get(), forward.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fft_c2c_out_out_Tensor_self_Tensor_dim_IntArrayRef_normalization_int64_t_forward_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchint64_t normalization, XPtrTorchbool forward) {
  auto r_out = lantern__fft_c2c_out_tensor_tensor_intarrayref_intt_bool(out.get(), self.get(), dim.get(), normalization.get(), forward.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace__cufft_get_plan_cache_size_device_index_int64_t (XPtrTorchint64_t device_index) {
  auto r_out = lantern__cufft_get_plan_cache_size_intt(device_index.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace__cufft_get_plan_cache_max_size_device_index_int64_t (XPtrTorchint64_t device_index) {
  auto r_out = lantern__cufft_get_plan_cache_max_size_intt(device_index.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_set_plan_cache_max_size_device_index_int64_t_max_size_int64_t (XPtrTorchint64_t device_index, XPtrTorchint64_t max_size) {
  lantern__cufft_set_plan_cache_max_size_intt_intt(device_index.get(), max_size.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_clear_plan_cache_device_index_int64_t (XPtrTorchint64_t device_index) {
  lantern__cufft_clear_plan_cache_intt(device_index.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_self_Tensor_indices_constc10Listc10optionalTensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices) {
  auto r_out = lantern_index_tensor_constclistcoptionaltensor(self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_index_copy_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_index_copy_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_put__self_Tensor_indices_constc10Listc10optionalTensor_values_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices, XPtrTorchTensor values, XPtrTorchbool accumulate) {
  auto r_out = lantern_index_put__tensor_constclistcoptionaltensor_tensor_bool(self.get(), indices.get(), values.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_put_self_Tensor_indices_constc10Listc10optionalTensor_values_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices, XPtrTorchTensor values, XPtrTorchbool accumulate) {
  auto r_out = lantern_index_put_tensor_constclistcoptionaltensor_tensor_bool(self.get(), indices.get(), values.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__index_put_impl__self_Tensor_indices_constc10Listc10optionalTensor_values_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexTensorList indices, XPtrTorchTensor values, XPtrTorchbool accumulate, XPtrTorchbool unsafe) {
  auto r_out = lantern__index_put_impl__tensor_constclistcoptionaltensor_tensor_bool_bool(self.get(), indices.get(), values.get(), accumulate.get(), unsafe.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_instance_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_use_input_stats_bool_momentum_double_eps_double_cudnn_enabled_bool (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool use_input_stats, XPtrTorchdouble momentum, XPtrTorchdouble eps, XPtrTorchbool cudnn_enabled) {
  auto r_out = lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), use_input_stats.get(), momentum.get(), eps.get(), cudnn_enabled.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inverse_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_inverse_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inverse_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_inverse_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__inverse_helper_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__inverse_helper_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isclose_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchdouble rtol, XPtrTorchdouble atol, XPtrTorchbool equal_nan) {
  auto r_out = lantern_isclose_tensor_tensor_double_double_bool(self.get(), other.get(), rtol.get(), atol.get(), equal_nan.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_out_out_Tensor_elements_Tensor_test_elements_Tensor (XPtrTorchTensor out, XPtrTorchTensor elements, XPtrTorchTensor test_elements, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_out_tensor_tensor_tensor_bool_bool(out.get(), elements.get(), test_elements.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_elements_Tensor_test_elements_Tensor (XPtrTorchTensor elements, XPtrTorchTensor test_elements, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_tensor_tensor_bool_bool(elements.get(), test_elements.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_out_out_Tensor_elements_Tensor_test_element_Scalar (XPtrTorchTensor out, XPtrTorchTensor elements, XPtrTorchScalar test_element, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_out_tensor_tensor_scalar_bool_bool(out.get(), elements.get(), test_element.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_elements_Tensor_test_element_Scalar (XPtrTorchTensor elements, XPtrTorchScalar test_element, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_tensor_scalar_bool_bool(elements.get(), test_element.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_out_out_Tensor_element_Scalar_test_elements_Tensor (XPtrTorchTensor out, XPtrTorchScalar element, XPtrTorchTensor test_elements, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_out_tensor_scalar_tensor_bool_bool(out.get(), element.get(), test_elements.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isin_element_Scalar_test_elements_Tensor (XPtrTorchScalar element, XPtrTorchTensor test_elements, XPtrTorchbool assume_unique, XPtrTorchbool invert) {
  auto r_out = lantern_isin_scalar_tensor_bool_bool(element.get(), test_elements.get(), assume_unique.get(), invert.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isnan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isnan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_distributed_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_distributed_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_floating_point_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_floating_point_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_complex_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_complex_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_conj_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_conj_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_neg_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isreal_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isreal_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_nonzero_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_nonzero_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_same_size_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_is_same_size_tensor_tensor(self.get(), other.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_signed_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_signed_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_is_inference_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_is_inference_tensor(self.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kl_div_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchbool log_target) {
  auto r_out = lantern_kl_div_tensor_tensor_intt_bool(self.get(), target.get(), reduction.get(), log_target.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kl_div_backward_grad_output_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchbool log_target) {
  auto r_out = lantern_kl_div_backward_tensor_tensor_tensor_intt_bool(grad_output.get(), self.get(), target.get(), reduction.get(), log_target.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kron_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_kron_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_kron_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_kron_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t_dim_int64_t (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_kthvalue_tensor_intt_intt_bool(self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool(values.get(), indices.get(), self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t_dim_Dimname (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_kthvalue_tensor_intt_dimname_bool(self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool(values.get(), indices.get(), self.get(), k.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_layer_norm_input_Tensor_normalized_shape_IntArrayRef (XPtrTorchTensor input, XPtrTorchIntArrayRef normalized_shape, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchdouble eps, XPtrTorchbool cudnn_enable) {
  auto r_out = lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(input.get(), normalized_shape.get(), weight.get(), bias.get(), eps.get(), cudnn_enable.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_input_Tensor_normalized_shape_IntArrayRef_weight_Tensor_bias_Tensor_eps_double (XPtrTorchTensor input, XPtrTorchIntArrayRef normalized_shape, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchdouble eps) {
  auto r_out = lantern_native_layer_norm_tensor_intarrayref_tensor_tensor_double(input.get(), normalized_shape.get(), weight.get(), bias.get(), eps.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_backward_grad_out_Tensor_input_Tensor_normalized_shape_IntArrayRef_mean_Tensor_rstd_Tensor_weight_Tensor_bias_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor grad_out, XPtrTorchTensor input, XPtrTorchIntArrayRef normalized_shape, XPtrTorchTensor mean, XPtrTorchTensor rstd, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, std::vector<bool> output_mask) {
  auto r_out = lantern_native_layer_norm_backward_tensor_tensor_intarrayref_tensor_tensor_tensor_tensor_stdarraybool(grad_out.get(), input.get(), normalized_shape.get(), mean.get(), rstd.get(), weight.get(), bias.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nan_to_num_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble nan, XPtrTorchOptionaldouble posinf, XPtrTorchOptionaldouble neginf) {
  auto r_out = lantern_nan_to_num_tensor_double_double_double(self.get(), nan.get(), posinf.get(), neginf.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nan_to_num__self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble nan, XPtrTorchOptionaldouble posinf, XPtrTorchOptionaldouble neginf) {
  auto r_out = lantern_nan_to_num__tensor_double_double_double(self.get(), nan.get(), posinf.get(), neginf.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nan_to_num_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionaldouble nan, XPtrTorchOptionaldouble posinf, XPtrTorchOptionaldouble neginf) {
  auto r_out = lantern_nan_to_num_out_tensor_tensor_double_double_double(out.get(), self.get(), nan.get(), posinf.get(), neginf.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linear_input_Tensor_weight_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias) {
  auto r_out = lantern_linear_tensor_tensor_tensor(input.get(), weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linear_out_out_Tensor_input_Tensor_weight_Tensor (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias) {
  auto r_out = lantern_linear_out_tensor_tensor_tensor_tensor(out.get(), input.get(), weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_linear_self_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias) {
  auto r_out = lantern_mkldnn_linear_tensor_tensor_tensor(self.get(), weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_linear_backward_input_input_size_IntArrayRef_grad_output_Tensor_weight_Tensor (XPtrTorchIntArrayRef input_size, XPtrTorchTensor grad_output, XPtrTorchTensor weight) {
  auto r_out = lantern_mkldnn_linear_backward_input_intarrayref_tensor_tensor(input_size.get(), grad_output.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_linear_backward_weights_grad_output_Tensor_input_Tensor_weight_Tensor_bias_defined_bool (XPtrTorchTensor grad_output, XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchbool bias_defined) {
  auto r_out = lantern_mkldnn_linear_backward_weights_tensor_tensor_tensor_bool(grad_output.get(), input.get(), weight.get(), bias_defined.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_linear_backward_self_Tensor_grad_output_Tensor_weight_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, std::vector<bool> output_mask) {
  auto r_out = lantern_mkldnn_linear_backward_tensor_tensor_tensor_stdarraybool(self.get(), grad_output.get(), weight.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_int8_weight_fp32_activation_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchTensor packed, XPtrTorchTensor col_offsets, XPtrTorchScalar weight_scale, XPtrTorchScalar weight_zero_point, XPtrTorchTensor bias) {
  auto r_out = lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(input.get(), weight.get(), packed.get(), col_offsets.get(), weight_scale.get(), weight_zero_point.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_int8_weight_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchTensor packed, XPtrTorchTensor col_offsets, XPtrTorchScalar weight_scale, XPtrTorchScalar weight_zero_point, XPtrTorchTensor bias) {
  auto r_out = lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(input.get(), weight.get(), packed.get(), col_offsets.get(), weight_scale.get(), weight_zero_point.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fbgemm_linear_quantize_weight_input_Tensor (XPtrTorchTensor input) {
  auto r_out = lantern_fbgemm_linear_quantize_weight_tensor(input.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchdouble(lantern_vector_get(wrap.get(), 2)),XPtrTorchint64_t(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_gemm_matrix_fp16_input_Tensor (XPtrTorchTensor input) {
  auto r_out = lantern_fbgemm_pack_gemm_matrix_fp16_tensor(input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_fp16_weight_fp32_activation_input_Tensor_packed_weight_Tensor_bias_Tensor (XPtrTorchTensor input, XPtrTorchTensor packed_weight, XPtrTorchTensor bias) {
  auto r_out = lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(input.get(), packed_weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_linear_fp16_weight_input_Tensor_packed_weight_Tensor_bias_Tensor (XPtrTorchTensor input, XPtrTorchTensor packed_weight, XPtrTorchTensor bias) {
  auto r_out = lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(input.get(), packed_weight.get(), bias.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor (XPtrTorchTensor input) {
  auto r_out = lantern_fbgemm_pack_quantized_matrix_tensor(input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor_K_int64_t_FALSE_int64_t (XPtrTorchTensor input, XPtrTorchint64_t K, XPtrTorchint64_t False) {
  auto r_out = lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(input.get(), K.get(), False.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ldexp_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ldexp_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ldexp__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ldexp__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ldexp_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ldexp_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linspace_start_Scalar_end_Scalar (XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchoptional_int64_t steps, XPtrTorchTensorOptions options) {
  auto r_out = lantern_linspace_scalar_scalar_intt_tensoroptions(start.get(), end.get(), steps.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linspace_out_out_Tensor_start_Scalar_end_Scalar (XPtrTorchTensor out, XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchoptional_int64_t steps) {
  auto r_out = lantern_linspace_out_tensor_scalar_scalar_intt(out.get(), start.get(), end.get(), steps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_log_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log10_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log10__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log10_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_log10_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log1p_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log1p__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log1p_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_log1p_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log2_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log2__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_log2_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logaddexp_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logaddexp_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp2_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logaddexp2_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logaddexp2_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_logaddexp2_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_xlogy_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_xlogy_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_xlogy_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy__self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_xlogy__tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy__self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_xlogy__tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_xlogy_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_out_out_Tensor_self_Scalar_other_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_xlogy_out_tensor_scalar_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_xlogy_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_xlogy_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logdet_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_logdet_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logspace_start_Scalar_end_Scalar (XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchoptional_int64_t steps, XPtrTorchdouble base, XPtrTorchTensorOptions options) {
  auto r_out = lantern_logspace_scalar_scalar_intt_double_tensoroptions(start.get(), end.get(), steps.get(), base.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logspace_out_out_Tensor_start_Scalar_end_Scalar (XPtrTorchTensor out, XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchoptional_int64_t steps, XPtrTorchdouble base) {
  auto r_out = lantern_logspace_out_tensor_scalar_scalar_intt_double(out.get(), start.get(), end.get(), steps.get(), base.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_log_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_log_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__log_softmax_tensor_intt_bool(self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_out_out_Tensor_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__log_softmax_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__log_softmax_backward_data_tensor_tensor_intt_tensor(grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__log_softmax_backward_data_out_out_Tensor_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__log_softmax_backward_data_out_tensor_tensor_tensor_intt_tensor(out.get(), grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__logcumsumexp_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__logcumsumexp_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__logcumsumexp_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__logcumsumexp_out_tensor_tensor_intt(out.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_logcumsumexp_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_logcumsumexp_out_tensor_tensor_intt(out.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_logcumsumexp_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logcumsumexp_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_logcumsumexp_out_tensor_tensor_dimname(out.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_logsumexp_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_logsumexp_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_logsumexp_tensor_dimnamelist_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_DimnameList (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_margin_ranking_loss_input1_Tensor_input2_Tensor_target_Tensor (XPtrTorchTensor input1, XPtrTorchTensor input2, XPtrTorchTensor target, XPtrTorchdouble margin, XPtrTorchint64_t reduction) {
  auto r_out = lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(input1.get(), input2.get(), target.get(), margin.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matmul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_matmul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matmul_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_matmul_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_rank_self_Tensor_tol_double (XPtrTorchTensor self, XPtrTorchdouble tol, XPtrTorchbool symmetric) {
  auto r_out = lantern_matrix_rank_tensor_double_bool(self.get(), tol.get(), symmetric.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_rank_self_Tensor (XPtrTorchTensor self, XPtrTorchbool symmetric) {
  auto r_out = lantern_matrix_rank_tensor_bool(self.get(), symmetric.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_power_self_Tensor_n_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_matrix_power_tensor_intt(self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_power_out_out_Tensor_self_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_matrix_power_out_tensor_tensor_intt(out.get(), self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_exp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_matrix_exp_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_matrix_exp_backward_self_Tensor_grad_Tensor (XPtrTorchTensor self, XPtrTorchTensor grad) {
  auto r_out = lantern_matrix_exp_backward_tensor_tensor(self.get(), grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__aminmax_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__aminmax_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__aminmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern__aminmax_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_aminmax_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_aminmax_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_aminmax_out_min_Tensor_max_Tensor_self_Tensor (XPtrTorchTensor min, XPtrTorchTensor max, XPtrTorchTensor self, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_aminmax_out_tensor_tensor_tensor_intt_bool(min.get(), max.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__compute_linear_combination_input_Tensor_coefficients_Tensor (XPtrTorchTensor input, XPtrTorchTensor coefficients) {
  auto r_out = lantern__compute_linear_combination_tensor_tensor(input.get(), coefficients.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__compute_linear_combination_out_out_Tensor_input_Tensor_coefficients_Tensor (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchTensor coefficients) {
  auto r_out = lantern__compute_linear_combination_out_tensor_tensor_tensor(out.get(), input.get(), coefficients.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_max_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor max, XPtrTorchTensor max_values, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_max_out_tensor_tensor_tensor_intt_bool(max.get(), max_values.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_max_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor max, XPtrTorchTensor max_values, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_max_out_tensor_tensor_tensor_dimname_bool(max.get(), max_values.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_value_selecting_reduction_backward_grad_Tensor_dim_int64_t_indices_Tensor_sizes_IntArrayRef_keepdim_bool (XPtrTorchTensor grad, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef sizes, XPtrTorchbool keepdim) {
  auto r_out = lantern_value_selecting_reduction_backward_tensor_intt_tensor_intarrayref_bool(grad.get(), dim.get(), indices.get(), sizes.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amax_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_amax_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amax_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_amax_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool1d_with_indices_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool1d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool2d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool2d_backward_grad_output_Tensor_output_Tensor_input_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchTensor input, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(grad_output.get(), output.get(), input.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool3d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_max_pool3d_backward_grad_output_Tensor_output_Tensor_input_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchTensor input, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_mkldnn_max_pool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(grad_output.get(), output.get(), input.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_max_pool1d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_quantized_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_max_pool2d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_mean_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_mean_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_mean_tensor_dimnamelist_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_DimnameList (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanmean_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_nanmean_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanmean_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_nanmean_out_tensor_tensor_intarrayref_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_median_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_median_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_median_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_median_out_tensor_tensor_tensor_intt_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_median_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_median_out_tensor_tensor_tensor_dimname_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanmedian_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_nanmedian_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nanmedian_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanmedian_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nanmedian_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanmedian_out_tensor_tensor_tensor_intt_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nanmedian_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanmedian_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nanmedian_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanmedian_out_tensor_tensor_tensor_dimname_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_min_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor min, XPtrTorchTensor min_indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_min_out_tensor_tensor_tensor_intt_bool(min.get(), min_indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_min_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor min, XPtrTorchTensor min_indices, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_min_out_tensor_tensor_tensor_dimname_bool(min.get(), min_indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amin_self_Tensor (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_amin_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_amin_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_amin_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(self.get(), weight.get(), bias.get(), padding.get(), stride.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (XPtrTorchIntArrayRef self_size, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool bias_defined) {
  auto r_out = lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(self_size.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), bias_defined.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_weights_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool bias_defined) {
  auto r_out = lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), bias_defined.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, std::vector<bool> output_mask) {
  auto r_out = lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble exponential_average_factor, XPtrTorchdouble epsilon) {
  auto r_out = lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), exponential_average_factor.get(), epsilon.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double (XPtrTorchTensor input, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchOptionalTensor save_mean, XPtrTorchOptionalTensor save_var, XPtrTorchdouble epsilon) {
  auto r_out = lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(input.get(), grad_output.get(), weight.get(), running_mean.get(), running_var.get(), save_mean.get(), save_var.get(), epsilon.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), bias.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchIntArrayRef self_size, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self_size.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_bias_grad_output_Tensor (XPtrTorchTensor grad_output) {
  auto r_out = lantern_miopen_convolution_backward_bias_tensor(grad_output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), bias.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), output_padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self.get(), weight.get(), bias.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchIntArrayRef self_size, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(self_size.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_depthwise_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (XPtrTorchTensor self, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(self.get(), grad_output.get(), weight.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_miopen_depthwise_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (XPtrTorchIntArrayRef weight_size, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups, XPtrTorchbool benchmark, XPtrTorchbool deterministic) {
  auto r_out = lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(weight_size.get(), grad_output.get(), self.get(), padding.get(), stride.get(), dilation.get(), groups.get(), benchmark.get(), deterministic.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (XPtrTorchTensor input, XPtrTorchTensorList weight, XPtrTorchint64_t weight_stride0, XPtrTorchTensor hx, XPtrTorchOptionalTensor cx, XPtrTorchint64_t mode, XPtrTorchint64_t hidden_size, XPtrTorchint64_t num_layers, XPtrTorchbool batch_first, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchIntArrayRef batch_sizes, XPtrTorchOptionalTensor dropout_state) {
  auto r_out = lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(input.get(), weight.get(), weight_stride0.get(), hx.get(), cx.get(), mode.get(), hidden_size.get(), num_layers.get(), batch_first.get(), dropout.get(), train.get(), bidirectional.get(), batch_sizes.get(), dropout_state.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (XPtrTorchTensor input, XPtrTorchTensorList weight, XPtrTorchint64_t weight_stride0, XPtrTorchTensor weight_buf, XPtrTorchTensor hx, XPtrTorchOptionalTensor cx, XPtrTorchTensor output, XPtrTorchOptionalTensor grad_output, XPtrTorchOptionalTensor grad_hy, XPtrTorchOptionalTensor grad_cy, XPtrTorchint64_t mode, XPtrTorchint64_t hidden_size, XPtrTorchint64_t num_layers, XPtrTorchbool batch_first, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchIntArrayRef batch_sizes, XPtrTorchOptionalTensor dropout_state, XPtrTorchTensor reserve, std::vector<bool> output_mask) {
  auto r_out = lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(input.get(), weight.get(), weight_stride0.get(), weight_buf.get(), hx.get(), cx.get(), output.get(), grad_output.get(), grad_hy.get(), grad_cy.get(), mode.get(), hidden_size.get(), num_layers.get(), batch_first.get(), dropout.get(), train.get(), bidirectional.get(), batch_sizes.get(), dropout_state.get(), reserve.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensorList(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_mm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mm_out_out_Tensor_self_Tensor_mat2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_mm_out_tensor_tensor_tensor(out.get(), self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_mm_sparse_Tensor_dense_Tensor (XPtrTorchTensor sparse, XPtrTorchTensor dense) {
  auto r_out = lantern__sparse_mm_tensor_tensor(sparse.get(), dense.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sparse_matmul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern__sparse_sparse_matmul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_mask_helper_t_Tensor_mask_indices_Tensor (XPtrTorchTensor t, XPtrTorchTensor mask_indices) {
  auto r_out = lantern__sparse_mask_helper_tensor_tensor(t.get(), mask_indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_mode_tensor_intt_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_mode_out_tensor_tensor_tensor_intt_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_mode_tensor_dimname_bool(self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_mode_out_tensor_tensor_tensor_dimname_bool(values.get(), indices.get(), self.get(), dim.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_mul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_mul_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mul_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_mul_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_multiply_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_multiply_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multiply_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_multiply_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mv_self_Tensor_vec_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec) {
  auto r_out = lantern_mv_tensor_tensor(self.get(), vec.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mv_out_out_Tensor_self_Tensor_vec_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor vec) {
  auto r_out = lantern_mv_out_tensor_tensor_tensor(out.get(), self.get(), vec.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mvlgamma_out_out_Tensor_self_Tensor_p_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_mvlgamma_out_tensor_tensor_intt(out.get(), self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mvlgamma_self_Tensor_p_int64_t (XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_mvlgamma_tensor_intt(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_copy_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t length) {
  auto r_out = lantern_narrow_copy_tensor_intt_intt_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_copy_out_out_Tensor_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t length) {
  auto r_out = lantern_narrow_copy_out_tensor_tensor_intt_intt_intt(out.get(), self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t length) {
  auto r_out = lantern_narrow_tensor_intt_intt_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_narrow_self_Tensor_dim_int64_t_start_Tensor_length_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchTensor start, XPtrTorchint64_t length) {
  auto r_out = lantern_narrow_tensor_intt_tensor_intt(self.get(), dim.get(), start.get(), length.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble momentum, XPtrTorchdouble eps) {
  auto r_out = lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), momentum.get(), eps.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_out_out_Tensor_save_mean_Tensor_save_invstd_Tensor_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double (XPtrTorchTensor out, XPtrTorchTensor save_mean, XPtrTorchTensor save_invstd, XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchbool training, XPtrTorchdouble momentum, XPtrTorchdouble eps) {
  auto r_out = lantern_native_batch_norm_out_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_double(out.get(), save_mean.get(), save_invstd.get(), input.get(), weight.get(), bias.get(), running_mean.get(), running_var.get(), training.get(), momentum.get(), eps.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_stats_input_Tensor_eps_double (XPtrTorchTensor input, XPtrTorchdouble eps) {
  auto r_out = lantern_batch_norm_stats_tensor_double(input.get(), eps.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_elemt_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchdouble eps) {
  auto r_out = lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double(input.get(), weight.get(), bias.get(), mean.get(), invstd.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_elemt_out_out_Tensor_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchdouble eps) {
  auto r_out = lantern_batch_norm_elemt_out_tensor_tensor_tensor_tensor_tensor_tensor_double(out.get(), input.get(), weight.get(), bias.get(), mean.get(), invstd.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_count_int64_t (XPtrTorchTensor input, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchdouble momentum, XPtrTorchdouble eps, XPtrTorchint64_t count) {
  auto r_out = lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt(input.get(), mean.get(), invstd.get(), running_mean.get(), running_var.get(), momentum.get(), eps.get(), count.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_with_counts_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_counts_Tensor (XPtrTorchTensor input, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchdouble momentum, XPtrTorchdouble eps, XPtrTorchTensor counts) {
  auto r_out = lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_tensor(input.get(), mean.get(), invstd.get(), running_mean.get(), running_var.get(), momentum.get(), eps.get(), counts.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_backward_grad_out_Tensor_input_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_invstd_Tensor_train_bool_eps_double_output_mask_stdarraybool3 (XPtrTorchTensor grad_out, XPtrTorchTensor input, XPtrTorchOptionalTensor weight, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchOptionalTensor save_mean, XPtrTorchOptionalTensor save_invstd, XPtrTorchbool train, XPtrTorchdouble eps, std::vector<bool> output_mask) {
  auto r_out = lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(grad_out.get(), input.get(), weight.get(), running_mean.get(), running_var.get(), save_mean.get(), save_invstd.get(), train.get(), eps.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_backward_reduce_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_input_g_bool_weight_g_bool_bias_g_bool (XPtrTorchTensor grad_out, XPtrTorchTensor input, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchOptionalTensor weight, XPtrTorchbool input_g, XPtrTorchbool weight_g, XPtrTorchbool bias_g) {
  auto r_out = lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool(grad_out.get(), input.get(), mean.get(), invstd.get(), weight.get(), input_g.get(), weight_g.get(), bias_g.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_batch_norm_backward_elemt_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_mean_dy_Tensor_mean_dy_xmu_Tensor_count_Tensor (XPtrTorchTensor grad_out, XPtrTorchTensor input, XPtrTorchTensor mean, XPtrTorchTensor invstd, XPtrTorchOptionalTensor weight, XPtrTorchTensor mean_dy, XPtrTorchTensor mean_dy_xmu, XPtrTorchTensor count) {
  auto r_out = lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(grad_out.get(), input.get(), mean.get(), invstd.get(), weight.get(), mean_dy.get(), mean_dy_xmu.get(), count.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_update_stats_input_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double (XPtrTorchTensor input, XPtrTorchOptionalTensor running_mean, XPtrTorchOptionalTensor running_var, XPtrTorchdouble momentum) {
  auto r_out = lantern_batch_norm_update_stats_tensor_tensor_tensor_double(input.get(), running_mean.get(), running_var.get(), momentum.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_input_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor weight, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref_intarrayref(input.get(), weight.get(), bias.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__nnpack_spatial_convolution_backward_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_mask_stdarraybool3 (XPtrTorchTensor input, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding, std::vector<bool> output_mask) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool(input.get(), grad_output.get(), weight.get(), padding.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_backward_input_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchTensor grad_output, XPtrTorchTensor weight, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(input.get(), grad_output.get(), weight.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__nnpack_spatial_convolution_backward_weight_input_Tensor_weightsize_IntArrayRef_grad_output_Tensor_padding_IntArrayRef (XPtrTorchTensor input, XPtrTorchIntArrayRef weightsize, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(input.get(), weightsize.get(), grad_output.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_size_IntArrayRef_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_ones_intarrayref_dimnamelist_tensoroptions(size.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_ones_intarrayref_tensoroptions(size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_out_out_Tensor_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_ones_out_tensor_intarrayref(out.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ones_like_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_ones_like_tensor_tensoroptions_memoryformat(self.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pairwise_distance_x1_Tensor_x2_Tensor (XPtrTorchTensor x1, XPtrTorchTensor x2, XPtrTorchdouble p, XPtrTorchdouble eps, XPtrTorchbool keepdim) {
  auto r_out = lantern_pairwise_distance_tensor_tensor_double_double_bool(x1.get(), x2.get(), p.get(), eps.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cdist_x1_Tensor_x2_Tensor (XPtrTorchTensor x1, XPtrTorchTensor x2, XPtrTorchdouble p, XPtrTorchoptional_int64_t compute_mode) {
  auto r_out = lantern_cdist_tensor_tensor_double_intt(x1.get(), x2.get(), p.get(), compute_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__euclidean_dist_x1_Tensor_x2_Tensor (XPtrTorchTensor x1, XPtrTorchTensor x2) {
  auto r_out = lantern__euclidean_dist_tensor_tensor(x1.get(), x2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cdist_forward_x1_Tensor_x2_Tensor_p_double_compute_mode_int64_t (XPtrTorchTensor x1, XPtrTorchTensor x2, XPtrTorchdouble p, XPtrTorchoptional_int64_t compute_mode) {
  auto r_out = lantern__cdist_forward_tensor_tensor_double_intt(x1.get(), x2.get(), p.get(), compute_mode.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cdist_backward_grad_Tensor_x1_Tensor_x2_Tensor_p_double_cdist_Tensor (XPtrTorchTensor grad, XPtrTorchTensor x1, XPtrTorchTensor x2, XPtrTorchdouble p, XPtrTorchTensor cdist) {
  auto r_out = lantern__cdist_backward_tensor_tensor_tensor_double_tensor(grad.get(), x1.get(), x2.get(), p.get(), cdist.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pdist_self_Tensor (XPtrTorchTensor self, XPtrTorchdouble p) {
  auto r_out = lantern_pdist_tensor_double(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pdist_forward_self_Tensor (XPtrTorchTensor self, XPtrTorchdouble p) {
  auto r_out = lantern__pdist_forward_tensor_double(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pdist_backward_grad_Tensor_self_Tensor_p_double_pdist_Tensor (XPtrTorchTensor grad, XPtrTorchTensor self, XPtrTorchdouble p, XPtrTorchTensor pdist) {
  auto r_out = lantern__pdist_backward_tensor_tensor_double_tensor(grad.get(), self.get(), p.get(), pdist.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cosine_similarity_x1_Tensor_x2_Tensor (XPtrTorchTensor x1, XPtrTorchTensor x2, XPtrTorchindex_int64_t dim, XPtrTorchdouble eps) {
  auto r_out = lantern_cosine_similarity_tensor_tensor_intt_double(x1.get(), x2.get(), dim.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_permute_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_permute_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_movedim_self_Tensor_source_IntArrayRef_destination_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef source, XPtrTorchIntArrayRef destination) {
  auto r_out = lantern_movedim_tensor_intarrayref_intarrayref(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_movedim_self_Tensor_source_int64_t_destination_int64_t (XPtrTorchTensor self, XPtrTorchint64_t source, XPtrTorchint64_t destination) {
  auto r_out = lantern_movedim_tensor_intt_intt(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_moveaxis_self_Tensor_source_IntArrayRef_destination_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef source, XPtrTorchIntArrayRef destination) {
  auto r_out = lantern_moveaxis_tensor_intarrayref_intarrayref(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_moveaxis_self_Tensor_source_int64_t_destination_int64_t (XPtrTorchTensor self, XPtrTorchint64_t source, XPtrTorchint64_t destination) {
  auto r_out = lantern_moveaxis_tensor_intt_intt(self.get(), source.get(), destination.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pixel_shuffle_self_Tensor_upscale_factor_int64_t (XPtrTorchTensor self, XPtrTorchint64_t upscale_factor) {
  auto r_out = lantern_pixel_shuffle_tensor_intt(self.get(), upscale_factor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pixel_unshuffle_self_Tensor_downscale_factor_int64_t (XPtrTorchTensor self, XPtrTorchint64_t downscale_factor) {
  auto r_out = lantern_pixel_unshuffle_tensor_intt(self.get(), downscale_factor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_channel_shuffle_self_Tensor_groups_int64_t (XPtrTorchTensor self, XPtrTorchint64_t groups) {
  auto r_out = lantern_channel_shuffle_tensor_intt(self.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pin_memory_self_Tensor (XPtrTorchTensor self, XPtrTorchDevice device) {
  auto r_out = lantern__pin_memory_tensor_device(self.get(), device.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pinverse_self_Tensor (XPtrTorchTensor self, XPtrTorchdouble rcond) {
  auto r_out = lantern_pinverse_tensor_double(self.get(), rcond.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_poisson_nll_loss_input_Tensor_target_Tensor_log_input_bool_full_bool_eps_double_reduction_int64_t (XPtrTorchTensor input, XPtrTorchTensor target, XPtrTorchbool log_input, XPtrTorchbool full, XPtrTorchdouble eps, XPtrTorchint64_t reduction) {
  auto r_out = lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(input.get(), target.get(), log_input.get(), full.get(), eps.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_rad2deg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_rad2deg__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rad2deg_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_rad2deg_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_deg2rad_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_deg2rad__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_deg2rad_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_deg2rad_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scalar_tensor_s_Scalar (XPtrTorchScalar s, XPtrTorchTensorOptions options) {
  auto r_out = lantern_scalar_tensor_scalar_tensoroptions(s.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_rand_intarrayref_dimnamelist_tensoroptions(size.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(size.get(), generator.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_rand_intarrayref_tensoroptions(size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator (XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchTensorOptions options) {
  auto r_out = lantern_rand_intarrayref_generator_tensoroptions(size.get(), generator.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_rand_out_tensor_intarrayref(out.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef_generator_Generator (XPtrTorchTensor out, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rand_out_tensor_intarrayref_generator(out.get(), size.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rand_like_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_rand_like_tensor_tensoroptions_memoryformat(self.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef (XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randint_intt_intarrayref_tensoroptions(high.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef_generator_Generator (XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randint_intt_intarrayref_generator_tensoroptions(high.get(), size.get(), generator.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef (XPtrTorchint64_t low, XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randint_intt_intt_intarrayref_tensoroptions(low.get(), high.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (XPtrTorchint64_t low, XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randint_intt_intt_intarrayref_generator_tensoroptions(low.get(), high.get(), size.get(), generator.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchint64_t high, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_randint_out_tensor_intt_intarrayref(out.get(), high.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef_generator_Generator (XPtrTorchTensor out, XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_randint_out_tensor_intt_intarrayref_generator(out.get(), high.get(), size.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchint64_t low, XPtrTorchint64_t high, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_randint_out_tensor_intt_intt_intarrayref(out.get(), low.get(), high.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (XPtrTorchTensor out, XPtrTorchint64_t low, XPtrTorchint64_t high, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_randint_out_tensor_intt_intt_intarrayref_generator(out.get(), low.get(), high.get(), size.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_like_self_Tensor_high_int64_t (XPtrTorchTensor self, XPtrTorchint64_t high, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_randint_like_tensor_intt_tensoroptions_memoryformat(self.get(), high.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randint_like_self_Tensor_low_int64_t_high_int64_t (XPtrTorchTensor self, XPtrTorchint64_t low, XPtrTorchint64_t high, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_randint_like_tensor_intt_intt_tensoroptions_memoryformat(self.get(), low.get(), high.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randn_intarrayref_tensoroptions(size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator (XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randn_intarrayref_generator_tensoroptions(size.get(), generator.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randn_intarrayref_dimnamelist_tensoroptions(size.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(size.get(), generator.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_randn_out_tensor_intarrayref(out.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef_generator_Generator (XPtrTorchTensor out, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_randn_out_tensor_intarrayref_generator(out.get(), size.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randn_like_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_randn_like_tensor_tensoroptions_memoryformat(self.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_n_int64_t (XPtrTorchint64_t n, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randperm_intt_tensoroptions(n.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_n_int64_t_generator_Generator (XPtrTorchint64_t n, XPtrTorchOptionalGenerator generator, XPtrTorchTensorOptions options) {
  auto r_out = lantern_randperm_intt_generator_tensoroptions(n.get(), generator.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchint64_t n) {
  auto r_out = lantern_randperm_out_tensor_intt(out.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t_generator_Generator (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_randperm_out_tensor_intt_generator(out.get(), n.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_range_start_Scalar_end_Scalar (XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchScalar step, XPtrTorchTensorOptions options) {
  auto r_out = lantern_range_scalar_scalar_scalar_tensoroptions(start.get(), end.get(), step.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_range_out_out_Tensor_start_Scalar_end_Scalar (XPtrTorchTensor out, XPtrTorchScalar start, XPtrTorchScalar end, XPtrTorchScalar step) {
  auto r_out = lantern_range_out_tensor_scalar_scalar_scalar(out.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ravel_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_ravel_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_reciprocal_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_reciprocal__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reciprocal_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_reciprocal_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_neg_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_neg__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_neg_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_neg_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_negative_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_negative__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_negative_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_negative_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_repeats_Tensor (XPtrTorchTensor repeats, XPtrTorchoptional_int64_t output_size) {
  auto r_out = lantern_repeat_interleave_tensor_intt(repeats.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_Tensor (XPtrTorchTensor self, XPtrTorchTensor repeats, XPtrTorchoptional_index_int64_t dim, XPtrTorchoptional_int64_t output_size) {
  auto r_out = lantern_repeat_interleave_tensor_tensor_intt_intt(self.get(), repeats.get(), dim.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_int64_t (XPtrTorchTensor self, XPtrTorchint64_t repeats, XPtrTorchoptional_index_int64_t dim, XPtrTorchoptional_int64_t output_size) {
  auto r_out = lantern_repeat_interleave_tensor_intt_intt_intt(self.get(), repeats.get(), dim.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reshape_self_Tensor_shape_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef shape) {
  auto r_out = lantern_reshape_tensor_intarrayref(self.get(), shape.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__reshape_alias_self_Tensor_size_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern__reshape_alias_tensor_intarrayref_intarrayref(self.get(), size.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_reshape_self_Tensor_shape_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef shape) {
  auto r_out = lantern__mkldnn_reshape_tensor_intarrayref(self.get(), shape.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_round_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_round__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_round_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_round_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rrelu_tensor_scalar_scalar_bool_generator(self.get(), lower.get(), upper.get(), training.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu__self_Tensor (XPtrTorchTensor self, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rrelu__tensor_scalar_scalar_bool_generator(self.get(), lower.get(), upper.get(), training.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_relu_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_relu__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu6_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_relu6_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_relu6__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_relu6__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prelu_self_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor weight) {
  auto r_out = lantern_prelu_tensor_tensor(self.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight) {
  auto r_out = lantern_prelu_backward_tensor_tensor_tensor(grad_output.get(), self.get(), weight.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_gelu_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_gelu_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_backward_out_grad_input_Tensor_grad_Tensor_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad, XPtrTorchTensor self) {
  auto r_out = lantern_gelu_backward_out_tensor_tensor_tensor(grad_input.get(), grad.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gelu_backward_grad_Tensor_self_Tensor (XPtrTorchTensor grad, XPtrTorchTensor self) {
  auto r_out = lantern_gelu_backward_tensor_tensor(grad.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_infinitely_differentiable_gelu_backward_grad_Tensor_self_Tensor (XPtrTorchTensor grad, XPtrTorchTensor self) {
  auto r_out = lantern_infinitely_differentiable_gelu_backward_tensor_tensor(grad.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_hardshrink_out_tensor_tensor_scalar(out.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_hardshrink_tensor_scalar(self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_backward_out_grad_input_Tensor_grad_out_Tensor_self_Tensor_lambd_Scalar (XPtrTorchTensor grad_input, XPtrTorchTensor grad_out, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_hardshrink_backward_out_tensor_tensor_tensor_scalar(grad_input.get(), grad_out.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (XPtrTorchTensor grad_out, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_hardshrink_backward_tensor_tensor_scalar(grad_out.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_rsqrt_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_rsqrt__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsqrt_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_rsqrt_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_self_Tensor_dim_Dimname_index_int64_t (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchindex_int64_t index) {
  auto r_out = lantern_select_tensor_dimname_intt(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_self_Tensor_dim_int64_t_index_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchindex_int64_t index) {
  auto r_out = lantern_select_tensor_intt_intt(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_select_backward_grad_output_Tensor_input_sizes_IntArrayRef_dim_int64_t_index_int64_t (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef input_sizes, XPtrTorchindex_int64_t dim, XPtrTorchindex_int64_t index) {
  auto r_out = lantern_select_backward_tensor_intarrayref_intt_intt(grad_output.get(), input_sizes.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_selu_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_selu_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_selu__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_selu__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_celu_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar alpha) {
  auto r_out = lantern_celu_tensor_scalar(self.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_celu__self_Tensor (XPtrTorchTensor self, XPtrTorchScalar alpha) {
  auto r_out = lantern_celu__tensor_scalar(self.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_silu_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_silu__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_silu_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_silu_backward_out_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_silu_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_silu_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mish_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_mish_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mish__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_mish__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mish_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_mish_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mish_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_mish_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sigmoid_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sigmoid__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sigmoid_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_logit_tensor_double(self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit__self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_logit__tensor_double(self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_logit_out_tensor_tensor_double(out.get(), self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sin_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sin__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sin_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sin_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sinc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sinc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sinc_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sinh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sinh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sinh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sinh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_detach_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_detach_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_detach__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_detach__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_size_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_size_tensor_intt(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_size_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_size_tensor_dimname(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slice_self_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_int64_t start, XPtrTorchoptional_int64_t end, XPtrTorchint64_t step) {
  auto r_out = lantern_slice_tensor_intt_intt_intt_intt(self.get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slice_backward_grad_output_Tensor_input_sizes_IntArrayRef_dim_int64_t_start_int64_t_end_int64_t_step_int64_t (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef input_sizes, XPtrTorchindex_int64_t dim, XPtrTorchint64_t start, XPtrTorchint64_t end, XPtrTorchint64_t step) {
  auto r_out = lantern_slice_backward_tensor_intarrayref_intt_intt_intt_intt(grad_output.get(), input_sizes.get(), dim.get(), start.get(), end.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slogdet_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_slogdet_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smm_self_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat2) {
  auto r_out = lantern_smm_tensor_tensor(self.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__softmax_tensor_intt_bool(self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_out_out_Tensor_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__softmax_out_tensor_tensor_intt_bool(out.get(), self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__softmax_backward_data_tensor_tensor_intt_tensor(grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__softmax_backward_data_out_grad_input_Tensor_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__softmax_backward_data_out_tensor_tensor_tensor_intt_tensor(grad_input.get(), grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unsafe_split_self_Tensor_split_size_int64_t (XPtrTorchTensor self, XPtrTorchint64_t split_size, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_unsafe_split_tensor_intt_intt(self.get(), split_size.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_split_self_Tensor_split_size_int64_t (XPtrTorchTensor self, XPtrTorchint64_t split_size, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_split_tensor_intt_intt(self.get(), split_size.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unsafe_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef split_sizes, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_unsafe_split_with_sizes_tensor_intarrayref_intt(self.get(), split_sizes.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef split_sizes, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_split_with_sizes_tensor_intarrayref_intt(self.get(), split_sizes.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_hsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_hsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_hsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_hsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_vsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_vsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_vsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_vsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_dsplit_self_Tensor_sections_int64_t (XPtrTorchTensor self, XPtrTorchint64_t sections) {
  auto r_out = lantern_dsplit_tensor_intt(self.get(), sections.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_dsplit_self_Tensor_indices_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef indices) {
  auto r_out = lantern_dsplit_tensor_intarrayref(self.get(), indices.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_squeeze_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_squeeze_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_squeeze_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_squeeze_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar(self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sspaddmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stack_tensors_TensorList (XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_stack_tensorlist_intt(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_stack_out_tensor_tensorlist_intt(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__stack_tensors_TensorList (XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__stack_tensorlist_intt(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__stack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__stack_out_tensor_tensorlist_intt(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hstack_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_hstack_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hstack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_hstack_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vstack_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_vstack_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vstack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_vstack_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dstack_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_dstack_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dstack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_dstack_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_stft_self_Tensor_n_fft_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n_fft, XPtrTorchoptional_int64_t hop_length, XPtrTorchoptional_int64_t win_length, XPtrTorchOptionalTensor window, XPtrTorchbool normalized, XPtrTorchoptional_bool onesided, XPtrTorchoptional_bool return_complex) {
  auto r_out = lantern_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(self.get(), n_fft.get(), hop_length.get(), win_length.get(), window.get(), normalized.get(), onesided.get(), return_complex.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_istft_self_Tensor_n_fft_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n_fft, XPtrTorchoptional_int64_t hop_length, XPtrTorchoptional_int64_t win_length, XPtrTorchOptionalTensor window, XPtrTorchbool center, XPtrTorchbool normalized, XPtrTorchoptional_bool onesided, XPtrTorchoptional_int64_t length, XPtrTorchbool return_complex) {
  auto r_out = lantern_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(self.get(), n_fft.get(), hop_length.get(), win_length.get(), window.get(), center.get(), normalized.get(), onesided.get(), length.get(), return_complex.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_stride_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_stride_tensor_intt(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_stride_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_stride_tensor_dimname(self.get(), dim.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_sum_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_sum_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_sum_tensor_dimnamelist_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_DimnameList (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_nansum_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_nansum_tensor_intarrayref_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nansum_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_nansum_out_tensor_tensor_intarrayref_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sqrt_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sqrt__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sqrt_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sqrt_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_square_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_square_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_square__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_square__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_square_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_square_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_std_tensor_bool(self.get(), unbiased.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_std_mean_tensor_bool(self.get(), unbiased.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_mean_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_mean_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_mean_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_mean_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_intarrayref_bool_bool(out.get(), self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_intarrayref_intt_bool(out.get(), self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_DimnameList (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_dimnamelist_bool_bool(out.get(), self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_std_out_tensor_tensor_dimnamelist_intt_bool(out.get(), self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_prod_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_prod_tensor_intt_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_prod_out_tensor_tensor_intt_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_prod_tensor_dimname_bool_scalartype(self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_prod_out_tensor_tensor_dimname_bool_scalartype(out.get(), self.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_t_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_t_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_tan_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_tan__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tan_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_tan_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_tanh_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_tanh__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_tanh_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tensordot_self_Tensor_other_Tensor_dims_self_IntArrayRef_dims_other_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchIndexIntArrayRef dims_self, XPtrTorchIndexIntArrayRef dims_other) {
  auto r_out = lantern_tensordot_tensor_tensor_intarrayref_intarrayref(self.get(), other.get(), dims_self.get(), dims_other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tensordot_out_out_Tensor_self_Tensor_other_Tensor_dims_self_IntArrayRef_dims_other_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchIndexIntArrayRef dims_self, XPtrTorchIndexIntArrayRef dims_other) {
  auto r_out = lantern_tensordot_out_tensor_tensor_tensor_intarrayref_intarrayref(out.get(), self.get(), other.get(), dims_self.get(), dims_other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_self_Tensor_threshold_Scalar_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar threshold, XPtrTorchScalar value) {
  auto r_out = lantern_threshold_tensor_scalar_scalar(self.get(), threshold.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold__self_Tensor_threshold_Scalar_value_Scalar (XPtrTorchTensor self, XPtrTorchScalar threshold, XPtrTorchScalar value) {
  auto r_out = lantern_threshold__tensor_scalar_scalar(self.get(), threshold.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_out_out_Tensor_self_Tensor_threshold_Scalar_value_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar threshold, XPtrTorchScalar value) {
  auto r_out = lantern_threshold_out_tensor_tensor_scalar_scalar(out.get(), self.get(), threshold.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_threshold_Scalar (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar threshold) {
  auto r_out = lantern_threshold_backward_out_tensor_tensor_tensor_scalar(grad_input.get(), grad_output.get(), self.get(), threshold.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_threshold_backward_grad_output_Tensor_self_Tensor_threshold_Scalar (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar threshold) {
  auto r_out = lantern_threshold_backward_tensor_tensor_scalar(grad_output.get(), self.get(), threshold.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tile_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_tile_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_transpose_tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim0, XPtrTorchDimname dim1) {
  auto r_out = lantern_transpose_tensor_dimname_dimname(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern__mkldnn_transpose_tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__mkldnn_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern__mkldnn_transpose__tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_one_hot_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t num_classes) {
  auto r_out = lantern_one_hot_tensor_intt(self.get(), num_classes.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flip_self_Tensor_dims_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_flip_tensor_intarrayref(self.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fliplr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_fliplr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flipud_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_flipud_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_roll_self_Tensor_shifts_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef shifts, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_roll_tensor_intarrayref_intarrayref(self.get(), shifts.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rot90_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchIndexIntArrayRef dims) {
  auto r_out = lantern_rot90_tensor_intt_intarrayref(self.get(), k.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapezoid_y_Tensor_x_Tensor (XPtrTorchTensor y, XPtrTorchTensor x, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_trapezoid_tensor_tensor_intt(y.get(), x.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapezoid_y_Tensor (XPtrTorchTensor y, XPtrTorchScalar dx, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_trapezoid_tensor_scalar_intt(y.get(), dx.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapz_y_Tensor_x_Tensor (XPtrTorchTensor y, XPtrTorchTensor x, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_trapz_tensor_tensor_intt(y.get(), x.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trapz_y_Tensor (XPtrTorchTensor y, XPtrTorchdouble dx, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_trapz_tensor_double_intt(y.get(), dx.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__trilinear_i1_Tensor_i2_Tensor_i3_Tensor_expand1_IntArrayRef_expand2_IntArrayRef_expand3_IntArrayRef_sumdim_IntArrayRef (XPtrTorchTensor i1, XPtrTorchTensor i2, XPtrTorchTensor i3, XPtrTorchIntArrayRef expand1, XPtrTorchIntArrayRef expand2, XPtrTorchIntArrayRef expand3, XPtrTorchIntArrayRef sumdim, XPtrTorchint64_t unroll_dim) {
  auto r_out = lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(i1.get(), i2.get(), i3.get(), expand1.get(), expand2.get(), expand3.get(), sumdim.get(), unroll_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triplet_margin_loss_anchor_Tensor_positive_Tensor_negative_Tensor (XPtrTorchTensor anchor, XPtrTorchTensor positive, XPtrTorchTensor negative, XPtrTorchdouble margin, XPtrTorchdouble p, XPtrTorchdouble eps, XPtrTorchbool swap, XPtrTorchint64_t reduction) {
  auto r_out = lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(anchor.get(), positive.get(), negative.get(), margin.get(), p.get(), eps.get(), swap.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_trunc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_trunc__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trunc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_trunc_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_fix_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_fix__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fix_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_fix_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace__has_compatible_shallow_copy_type_self_Tensor_from_Tensor (XPtrTorchTensor self, XPtrTorchTensor from) {
  auto r_out = lantern__has_compatible_shallow_copy_type_tensor_tensor(self.get(), from.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique_self_Tensor (XPtrTorchTensor self, XPtrTorchbool sorted, XPtrTorchbool return_inverse) {
  auto r_out = lantern__unique_tensor_bool_bool(self.get(), sorted.get(), return_inverse.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool sorted, XPtrTorchbool return_inverse, XPtrTorchbool return_counts) {
  auto r_out = lantern_unique_dim_tensor_intt_bool_bool_bool(self.get(), dim.get(), sorted.get(), return_inverse.get(), return_counts.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_consecutive_self_Tensor (XPtrTorchTensor self, XPtrTorchbool return_inverse, XPtrTorchbool return_counts, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_unique_consecutive_tensor_bool_bool_intt(self.get(), return_inverse.get(), return_counts.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_consecutive_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool return_inverse, XPtrTorchbool return_counts) {
  auto r_out = lantern_unique_dim_consecutive_tensor_intt_bool_bool(self.get(), dim.get(), return_inverse.get(), return_counts.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique2_self_Tensor (XPtrTorchTensor self, XPtrTorchbool sorted, XPtrTorchbool return_inverse, XPtrTorchbool return_counts) {
  auto r_out = lantern__unique2_tensor_bool_bool_bool(self.get(), sorted.get(), return_inverse.get(), return_counts.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__unsafe_view_self_Tensor_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef size) {
  auto r_out = lantern__unsafe_view_tensor_intarrayref(self.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_unsqueeze_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_unsqueeze_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_vander_x_Tensor (XPtrTorchTensor x, XPtrTorchoptional_int64_t False, XPtrTorchbool increasing) {
  auto r_out = lantern_vander_tensor_intt_bool(x.get(), False.get(), increasing.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_var_tensor_bool(self.get(), unbiased.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_intarrayref_bool_bool(out.get(), self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_intarrayref_intt_bool(out.get(), self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_DimnameList (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_dimnamelist_bool_bool(out.get(), self.get(), dim.get(), unbiased.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_out_tensor_tensor_dimnamelist_intt_bool(out.get(), self.get(), dim.get(), correction.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor (XPtrTorchTensor self, XPtrTorchbool unbiased) {
  auto r_out = lantern_var_mean_tensor_bool(self.get(), unbiased.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_mean_tensor_intarrayref_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_IntArrayRef_correction_int64_t (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_mean_tensor_intarrayref_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_DimnameList (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchbool unbiased, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_mean_tensor_dimnamelist_bool_bool(self.get(), dim.get(), unbiased.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_DimnameList_correction_int64_t (XPtrTorchTensor self, XPtrTorchDimnameList dim, XPtrTorchoptional_int64_t correction, XPtrTorchbool keepdim) {
  auto r_out = lantern_var_mean_tensor_dimnamelist_intt_bool(self.get(), dim.get(), correction.get(), keepdim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor condition, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_where_tensor_tensor_tensor(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Scalar_other_Tensor (XPtrTorchTensor condition, XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_where_tensor_scalar_tensor(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor condition, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_where_tensor_tensor_scalar(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_where_condition_Tensor_self_Scalar_other_Scalar (XPtrTorchTensor condition, XPtrTorchScalar self, XPtrTorchScalar other) {
  auto r_out = lantern_where_tensor_scalar_scalar(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_where_condition_Tensor (XPtrTorchTensor condition) {
  auto r_out = lantern_where_tensor(condition.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__s_where_condition_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor condition, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern__s_where_tensor_tensor_tensor(condition.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_except_dim_v_Tensor (XPtrTorchTensor v, XPtrTorchint64_t pow, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_norm_except_dim_tensor_intt_intt(v.get(), pow.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__weight_norm_v_Tensor_g_Tensor (XPtrTorchTensor v, XPtrTorchTensor g, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__weight_norm_tensor_tensor_intt(v.get(), g.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_v_Tensor_g_Tensor (XPtrTorchTensor v, XPtrTorchTensor g, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__weight_norm_cuda_interface_tensor_tensor_intt(v.get(), g.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (XPtrTorchTensor grad_w, XPtrTorchTensor saved_v, XPtrTorchTensor saved_g, XPtrTorchTensor saved_norms, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt(grad_w.get(), saved_v.get(), saved_g.get(), saved_norms.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_differentiable_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (XPtrTorchTensor grad_w, XPtrTorchTensor saved_v, XPtrTorchTensor saved_g, XPtrTorchTensor saved_norms, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt(grad_w.get(), saved_v.get(), saved_g.get(), saved_norms.get(), dim.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_size_IntArrayRef_names_DimnameList (XPtrTorchIntArrayRef size, XPtrTorchOptionalDimnameList names, XPtrTorchTensorOptions options) {
  auto r_out = lantern_zeros_intarrayref_dimnamelist_tensoroptions(size.get(), names.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_size_IntArrayRef (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_zeros_intarrayref_tensoroptions(size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_out_out_Tensor_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIntArrayRef size) {
  auto r_out = lantern_zeros_out_tensor_intarrayref(out.get(), size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zeros_like_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_zeros_like_tensor_tensoroptions_memoryformat(self.get(), options.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__standard_gamma_grad_self_Tensor_output_Tensor (XPtrTorchTensor self, XPtrTorchTensor output) {
  auto r_out = lantern__standard_gamma_grad_tensor_tensor(self.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__standard_gamma_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern__standard_gamma_tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__dirichlet_grad_x_Tensor_alpha_Tensor_total_Tensor (XPtrTorchTensor x, XPtrTorchTensor alpha, XPtrTorchTensor total) {
  auto r_out = lantern__dirichlet_grad_tensor_tensor_tensor(x.get(), alpha.get(), total.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sample_dirichlet_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern__sample_dirichlet_tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_poisson_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_poisson_tensor_generator(self.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_binomial_count_Tensor_prob_Tensor (XPtrTorchTensor count, XPtrTorchTensor prob, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_binomial_tensor_tensor_generator(count.get(), prob.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_native_norm_self_Tensor_p_Scalar (XPtrTorchTensor self, XPtrTorchScalar p) {
  auto r_out = lantern_native_norm_tensor_scalar(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_native_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_native_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__sparse_sum_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchDtype dtype) {
  auto r_out = lantern__sparse_sum_tensor_scalartype(self.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim) {
  auto r_out = lantern__sparse_sum_tensor_intarrayref(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchDtype dtype) {
  auto r_out = lantern__sparse_sum_tensor_intarrayref_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_sum_backward_grad_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor grad, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim) {
  auto r_out = lantern__sparse_sum_backward_tensor_tensor_intarrayref(grad.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern__sparse_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern__sparse_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__sparse_softmax_tensor_intt_bool(self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__sparse_softmax_backward_data_tensor_tensor_intt_tensor(grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern__sparse_log_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern__sparse_log_softmax_tensor_dimname_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_self_Tensor_dim_int64_t_half_to_float_bool (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool half_to_float) {
  auto r_out = lantern__sparse_log_softmax_tensor_intt_bool(self.get(), dim.get(), half_to_float.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_log_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output, XPtrTorchindex_int64_t dim, XPtrTorchTensor self) {
  auto r_out = lantern__sparse_log_softmax_backward_data_tensor_tensor_intt_tensor(grad_output.get(), output.get(), dim.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDtype dtype) {
  auto r_out = lantern_norm_tensor_scalar_scalartype(self.get(), p.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar (XPtrTorchTensor self, XPtrTorchScalar p) {
  auto r_out = lantern_norm_tensor_scalar(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_norm_tensor_scalar_intarrayref_bool(self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out.get(), self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(out.get(), self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_norm_tensor_scalar_dimnamelist_bool_scalartype(self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool (XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_norm_tensor_scalar_dimnamelist_bool(self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim, XPtrTorchDtype dtype) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(out.get(), self.get(), p.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar p, XPtrTorchDimnameList dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(out.get(), self.get(), p.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_frexp_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_frexp_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_frexp_out_mantissa_Tensor_exponent_Tensor_self_Tensor (XPtrTorchTensor mantissa, XPtrTorchTensor exponent, XPtrTorchTensor self) {
  auto r_out = lantern_frexp_out_tensor_tensor_tensor(mantissa.get(), exponent.get(), self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_frobenius_norm_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_frobenius_norm_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_frobenius_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_self_Tensor (XPtrTorchTensor self, XPtrTorchbool keepdim) {
  auto r_out = lantern_nuclear_norm_tensor_bool(self.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchbool keepdim) {
  auto r_out = lantern_nuclear_norm_out_tensor_tensor_bool(out.get(), self.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nuclear_norm_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_clone_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_clone_tensor_memoryformat(self.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_positive_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_positive_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_resize_as__self_Tensor_the_template_Tensor (XPtrTorchTensor self, XPtrTorchTensor the_template, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern_resize_as__tensor_tensor_memoryformat(self.get(), the_template.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_resize_as_sparse__self_Tensor_the_template_Tensor (XPtrTorchTensor self, XPtrTorchTensor the_template) {
  auto r_out = lantern_resize_as_sparse__tensor_tensor(self.get(), the_template.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_zero__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_zero__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_sub_out_tensor_tensor_tensor_scalar(out.get(), self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_sub_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sub_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_sub_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_subtract_out_tensor_tensor_tensor_scalar(out.get(), self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_subtract_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_subtract_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_subtract_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsub_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern_rsub_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_heaviside_out_out_Tensor_self_Tensor_values_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor values) {
  auto r_out = lantern_heaviside_out_tensor_tensor_tensor(out.get(), self.get(), values.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_heaviside_self_Tensor_values_Tensor (XPtrTorchTensor self, XPtrTorchTensor values) {
  auto r_out = lantern_heaviside_tensor_tensor(self.get(), values.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rsub_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other, XPtrTorchScalar alpha) {
  auto r_out = lantern_rsub_tensor_scalar_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_addmm_self_Tensor_sparse_Tensor_dense_Tensor (XPtrTorchTensor self, XPtrTorchTensor sparse, XPtrTorchTensor dense, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(self.get(), sparse.get(), dense.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor self, XPtrTorchTensor mat1, XPtrTorchTensor mat2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addmm_tensor_tensor_tensor_scalar_scalar(self.get(), mat1.get(), mat2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_csr_tensor_crow_indices_Tensor_col_indices_Tensor_values_Tensor_size_IntArrayRef_options_TensorOptions (XPtrTorchTensor crow_indices, XPtrTorchTensor col_indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_sparse_csr_tensor_tensor_tensor_tensor_intarrayref_tensoroptions(crow_indices.get(), col_indices.get(), values.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_csr_tensor_crow_indices_Tensor_col_indices_Tensor_values_Tensor_options_TensorOptions (XPtrTorchTensor crow_indices, XPtrTorchTensor col_indices, XPtrTorchTensor values, XPtrTorchTensorOptions options) {
  auto r_out = lantern_sparse_csr_tensor_tensor_tensor_tensor_tensoroptions(crow_indices.get(), col_indices.get(), values.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_csr_tensor_unsafe_crow_indices_Tensor_col_indices_Tensor_values_Tensor_size_IntArrayRef (XPtrTorchTensor crow_indices, XPtrTorchTensor col_indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern__sparse_csr_tensor_unsafe_tensor_tensor_tensor_intarrayref_tensoroptions(crow_indices.get(), col_indices.get(), values.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_size_IntArrayRef_options_TensorOptions (XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_sparse_coo_tensor_intarrayref_tensoroptions(size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor_options_TensorOptions (XPtrTorchIndexTensor indices, XPtrTorchTensor values, XPtrTorchTensorOptions options) {
  auto r_out = lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(indices.get(), values.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor_size_IntArrayRef_options_TensorOptions (XPtrTorchIndexTensor indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(indices.get(), values.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_unsafe_indices_Tensor_values_Tensor_size_IntArrayRef (XPtrTorchIndexTensor indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(indices.get(), values.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__validate_sparse_coo_tensor_args_indices_Tensor_values_Tensor_size_IntArrayRef (XPtrTorchIndexTensor indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size) {
  lantern__validate_sparse_coo_tensor_args_tensor_tensor_intarrayref(indices.get(), values.get(), size.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__validate_sparse_csr_tensor_args_crow_indices_Tensor_col_indices_Tensor_values_Tensor_size_IntArrayRef (XPtrTorchTensor crow_indices, XPtrTorchTensor col_indices, XPtrTorchTensor values, XPtrTorchIntArrayRef size) {
  lantern__validate_sparse_csr_tensor_args_tensor_tensor_tensor_intarrayref(crow_indices.get(), col_indices.get(), values.get(), size.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_with_dims_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_options_TensorOptions (XPtrTorchint64_t sparse_dim, XPtrTorchint64_t dense_dim, XPtrTorchIntArrayRef size, XPtrTorchTensorOptions options) {
  auto r_out = lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions(sparse_dim.get(), dense_dim.get(), size.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__sparse_coo_tensor_with_dims_and_tensors_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_indices_Tensor_values_Tensor_options_TensorOptions (XPtrTorchint64_t sparse_dim, XPtrTorchint64_t dense_dim, XPtrTorchIntArrayRef size, XPtrTorchIndexTensor indices, XPtrTorchTensor values, XPtrTorchTensorOptions options) {
  auto r_out = lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions(sparse_dim.get(), dense_dim.get(), size.get(), indices.get(), values.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__to_cpu_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__to_cpu_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_to_dense_backward_grad_Tensor_input_Tensor (XPtrTorchTensor grad, XPtrTorchTensor input) {
  auto r_out = lantern_to_dense_backward_tensor_tensor(grad.get(), input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__coalesce_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__coalesce_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hspmm_out_out_Tensor_mat1_Tensor_mat2_Tensor (XPtrTorchTensor out, XPtrTorchTensor mat1, XPtrTorchTensor mat2) {
  auto r_out = lantern_hspmm_out_tensor_tensor_tensor(out.get(), mat1.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hspmm_mat1_Tensor_mat2_Tensor (XPtrTorchTensor mat1, XPtrTorchTensor mat2) {
  auto r_out = lantern_hspmm_tensor_tensor(mat1.get(), mat2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_copy_sparse_to_sparse__self_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchTensor src, XPtrTorchbool non_blocking) {
  auto r_out = lantern_copy_sparse_to_sparse__tensor_tensor_bool(self.get(), src.get(), non_blocking.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unbind_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_unbind_tensor_intt(self.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unbind_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim) {
  auto r_out = lantern_unbind_tensor_dimname(self.get(), dim.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_reorder_conv2d_weight_self_Tensor (XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(self.get(), padding.get(), stride.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_reorder_conv3d_weight_self_Tensor (XPtrTorchTensor self, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef dilation, XPtrTorchint64_t groups) {
  auto r_out = lantern_mkldnn_reorder_conv3d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(self.get(), padding.get(), stride.get(), dilation.get(), groups.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_to_mkldnn_backward_grad_Tensor_input_Tensor (XPtrTorchTensor grad, XPtrTorchTensor input) {
  auto r_out = lantern_to_mkldnn_backward_tensor_tensor(grad.get(), input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantize_per_tensor_self_Tensor_scale_double_zero_point_int64_t_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchdouble scale, XPtrTorchint64_t zero_point, XPtrTorchDtype dtype) {
  auto r_out = lantern_quantize_per_tensor_tensor_double_intt_scalartype(self.get(), scale.get(), zero_point.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantize_per_tensor_self_Tensor_scale_Tensor_zero_point_Tensor_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchDtype dtype) {
  auto r_out = lantern_quantize_per_tensor_tensor_tensor_tensor_scalartype(self.get(), scale.get(), zero_point.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_quantize_per_tensor_tensors_TensorList_scales_Tensor_zero_points_Tensor_dtype_ScalarType (XPtrTorchTensorList tensors, XPtrTorchTensor scales, XPtrTorchTensor zero_points, XPtrTorchDtype dtype) {
  auto r_out = lantern_quantize_per_tensor_tensorlist_tensor_tensor_scalartype(tensors.get(), scales.get(), zero_points.get(), dtype.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantize_per_channel_self_Tensor_scales_Tensor_zero_points_Tensor_axis_int64_t_dtype_ScalarType (XPtrTorchTensor self, XPtrTorchTensor scales, XPtrTorchTensor zero_points, XPtrTorchint64_t axis, XPtrTorchDtype dtype) {
  auto r_out = lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype(self.get(), scales.get(), zero_points.get(), axis.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dequantize_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_dequantize_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_dequantize_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_dequantize_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchdouble cpp_torch_namespace_q_scale_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_q_scale_tensor(self.get());
return XPtrTorchdouble(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_q_zero_point_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_q_zero_point_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_q_per_channel_scales_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_q_per_channel_scales_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_q_per_channel_zero_points_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_q_per_channel_zero_points_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchint64_t cpp_torch_namespace_q_per_channel_axis_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_q_per_channel_axis_tensor(self.get());
return XPtrTorchint64_t(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_int_repr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_int_repr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__make_per_tensor_quantized_tensor_self_Tensor_scale_double_zero_point_int64_t (XPtrTorchTensor self, XPtrTorchdouble scale, XPtrTorchint64_t zero_point) {
  auto r_out = lantern__make_per_tensor_quantized_tensor_tensor_double_intt(self.get(), scale.get(), zero_point.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__make_per_channel_quantized_tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t axis) {
  auto r_out = lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt(self.get(), scale.get(), zero_point.get(), axis.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_tensor_affine_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchdouble scale, XPtrTorchint64_t zero_point, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt(self.get(), scale.get(), zero_point.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_tensor_affine_self_Tensor_scale_Tensor_zero_point_Tensor_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_tensor_tensor_tensor_intt_intt(self.get(), scale.get(), zero_point.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fake_quantize_per_tensor_affine_cachemask_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchdouble scale, XPtrTorchint64_t zero_point, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_cachemask_tensor_double_intt_intt_intt(self.get(), scale.get(), zero_point.get(), quant_min.get(), quant_max.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_self_Tensor_scale_Tensor_zero_point_Tensor_fake_quant_enabled_Tensor_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchTensor fake_quant_enabled, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_tensor_tensor_tensor_tensor_intt_intt(self.get(), scale.get(), zero_point.get(), fake_quant_enabled.get(), quant_min.get(), quant_max.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_tensor_affine_cachemask_backward_grad_Tensor_mask_Tensor (XPtrTorchTensor grad, XPtrTorchTensor mask) {
  auto r_out = lantern_fake_quantize_per_tensor_affine_cachemask_backward_tensor_tensor(grad.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fake_quantize_learnable_per_tensor_affine_self_Tensor_scale_Tensor_zero_point_Tensor_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchdouble grad_factor) {
  auto r_out = lantern__fake_quantize_learnable_per_tensor_affine_tensor_tensor_tensor_intt_intt_double(self.get(), scale.get(), zero_point.get(), quant_min.get(), quant_max.get(), grad_factor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fake_quantize_learnable_per_tensor_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor grad, XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchdouble grad_factor) {
  auto r_out = lantern__fake_quantize_learnable_per_tensor_affine_backward_tensor_tensor_tensor_tensor_intt_intt_double(grad.get(), self.get(), scale.get(), zero_point.get(), quant_min.get(), quant_max.get(), grad_factor.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_channel_affine_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t axis, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(self.get(), scale.get(), zero_point.get(), axis.get(), quant_min.get(), quant_max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fake_quantize_per_channel_affine_cachemask_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t axis, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max) {
  auto r_out = lantern_fake_quantize_per_channel_affine_cachemask_tensor_tensor_tensor_intt_intt_intt(self.get(), scale.get(), zero_point.get(), axis.get(), quant_min.get(), quant_max.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fake_quantize_per_channel_affine_cachemask_backward_grad_Tensor_mask_Tensor (XPtrTorchTensor grad, XPtrTorchTensor mask) {
  auto r_out = lantern_fake_quantize_per_channel_affine_cachemask_backward_tensor_tensor(grad.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__fake_quantize_learnable_per_channel_affine_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t axis, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchdouble grad_factor) {
  auto r_out = lantern__fake_quantize_learnable_per_channel_affine_tensor_tensor_tensor_intt_intt_intt_double(self.get(), scale.get(), zero_point.get(), axis.get(), quant_min.get(), quant_max.get(), grad_factor.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fake_quantize_learnable_per_channel_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (XPtrTorchTensor grad, XPtrTorchTensor self, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchint64_t axis, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchdouble grad_factor) {
  auto r_out = lantern__fake_quantize_learnable_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt_double(grad.get(), self.get(), scale.get(), zero_point.get(), axis.get(), quant_min.get(), quant_max.get(), grad_factor.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fused_moving_avg_obs_fake_quant_self_Tensor_observer_on_Tensor_fake_quant_on_Tensor_running_min_Tensor_running_max_Tensor_scale_Tensor_zero_point_Tensor_averaging_const_double_quant_min_int64_t_quant_max_int64_t_ch_axis_int64_t (XPtrTorchTensor self, XPtrTorchTensor observer_on, XPtrTorchTensor fake_quant_on, XPtrTorchTensor running_min, XPtrTorchTensor running_max, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchdouble averaging_const, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchint64_t ch_axis, XPtrTorchbool per_row_fake_quant, XPtrTorchbool symmetric_quant) {
  auto r_out = lantern_fused_moving_avg_obs_fake_quant_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_intt_intt_intt_bool_bool(self.get(), observer_on.get(), fake_quant_on.get(), running_min.get(), running_max.get(), scale.get(), zero_point.get(), averaging_const.get(), quant_min.get(), quant_max.get(), ch_axis.get(), per_row_fake_quant.get(), symmetric_quant.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fused_moving_avg_obs_fq_helper_self_Tensor_observer_on_Tensor_fake_quant_on_Tensor_running_min_Tensor_running_max_Tensor_scale_Tensor_zero_point_Tensor_averaging_const_double_quant_min_int64_t_quant_max_int64_t_ch_axis_int64_t (XPtrTorchTensor self, XPtrTorchTensor observer_on, XPtrTorchTensor fake_quant_on, XPtrTorchTensor running_min, XPtrTorchTensor running_max, XPtrTorchTensor scale, XPtrTorchTensor zero_point, XPtrTorchdouble averaging_const, XPtrTorchint64_t quant_min, XPtrTorchint64_t quant_max, XPtrTorchint64_t ch_axis, XPtrTorchbool per_row_fake_quant, XPtrTorchbool symmetric_quant) {
  auto r_out = lantern__fused_moving_avg_obs_fq_helper_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_intt_intt_intt_bool_bool(self.get(), observer_on.get(), fake_quant_on.get(), running_min.get(), running_max.get(), scale.get(), zero_point.get(), averaging_const.get(), quant_min.get(), quant_max.get(), ch_axis.get(), per_row_fake_quant.get(), symmetric_quant.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__choose_qparams_per_tensor_self_Tensor (XPtrTorchTensor self, XPtrTorchbool reduce_range) {
  auto r_out = lantern__choose_qparams_per_tensor_tensor_bool(self.get(), reduce_range.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchdouble(lantern_vector_get(wrap.get(), 0)),XPtrTorchint64_t(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__saturate_weight_to_fp16_weight_Tensor (XPtrTorchTensor weight) {
  auto r_out = lantern__saturate_weight_to_fp16_tensor(weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_choose_qparams_optimized_input_Tensor_numel_int64_t_n_bins_int64_t_ratio_double_bit_width_int64_t (XPtrTorchTensor input, XPtrTorchint64_t numel, XPtrTorchint64_t n_bins, XPtrTorchdouble ratio, XPtrTorchint64_t bit_width) {
  auto r_out = lantern_choose_qparams_optimized_tensor_intt_intt_double_intt(input.get(), numel.get(), n_bins.get(), ratio.get(), bit_width.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__to_copy_self_Tensor (XPtrTorchTensor self, XPtrTorchTensorOptions options, XPtrTorchbool non_blocking, XPtrTorchoptional_memory_format memory_format) {
  auto r_out = lantern__to_copy_tensor_tensoroptions_bool_memoryformat(self.get(), options.get(), non_blocking.get(), memory_format.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_meshgrid_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_meshgrid_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_meshgrid_tensors_TensorList_indexing_c10string_view (XPtrTorchTensorList tensors, XPtrTorchstring_view indexing) {
  auto r_out = lantern_meshgrid_tensorlist_cstringview(tensors.get(), indexing.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cartesian_prod_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_cartesian_prod_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_combinations_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t r, XPtrTorchbool with_replacement) {
  auto r_out = lantern_combinations_tensor_intt_bool(self.get(), r.get(), with_replacement.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalarType cpp_torch_namespace_result_type_other_Tensor_tensor_Tensor (XPtrTorchTensor tensor, XPtrTorchTensor other) {
  auto r_out = lantern_result_type_tensor_tensor(tensor.get(), other.get());
return XPtrTorchScalarType(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalarType cpp_torch_namespace_result_type_other_Scalar_tensor_Tensor (XPtrTorchTensor tensor, XPtrTorchScalar other) {
  auto r_out = lantern_result_type_tensor_scalar(tensor.get(), other.get());
return XPtrTorchScalarType(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalarType cpp_torch_namespace_result_type_scalar_Scalar_tensor_Tensor (XPtrTorchScalar scalar, XPtrTorchTensor tensor) {
  auto r_out = lantern_result_type_scalar_tensor(scalar.get(), tensor.get());
return XPtrTorchScalarType(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalarType cpp_torch_namespace_result_type_scalar1_Scalar_scalar2_Scalar (XPtrTorchScalar scalar1, XPtrTorchScalar scalar2) {
  auto r_out = lantern_result_type_scalar_scalar(scalar1.get(), scalar2.get());
return XPtrTorchScalarType(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_can_cast_from_ScalarType_to_ScalarType (XPtrTorchDtype from, XPtrTorchDtype to) {
  auto r_out = lantern_can_cast_scalartype_scalartype(from.get(), to.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalarType cpp_torch_namespace_promote_types_type1_ScalarType_type2_ScalarType (XPtrTorchDtype type1, XPtrTorchDtype type2) {
  auto r_out = lantern_promote_types_scalartype_scalartype(type1.get(), type2.get());
return XPtrTorchScalarType(r_out);
}

// [[Rcpp::export]]
XPtrTorchScalar cpp_torch_namespace__local_scalar_dense_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__local_scalar_dense_tensor(self.get());
return XPtrTorchScalar(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_input_gates_Tensor_hidden_gates_Tensor_cx_Tensor (XPtrTorchTensor input_gates, XPtrTorchTensor hidden_gates, XPtrTorchTensor cx, XPtrTorchOptionalTensor input_bias, XPtrTorchOptionalTensor hidden_bias) {
  auto r_out = lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor(input_gates.get(), hidden_gates.get(), cx.get(), input_bias.get(), hidden_bias.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_cx_Tensor_cy_Tensor_workspace_Tensor_has_bias_bool (XPtrTorchOptionalTensor grad_hy, XPtrTorchOptionalTensor grad_cy, XPtrTorchTensor cx, XPtrTorchTensor cy, XPtrTorchTensor workspace, XPtrTorchbool has_bias) {
  auto r_out = lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool(grad_hy.get(), grad_cy.get(), cx.get(), cy.get(), workspace.get(), has_bias.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_input_gates_Tensor_hidden_gates_Tensor_input_bias_Tensor_hidden_bias_Tensor_cx_Tensor_cy_Tensor (XPtrTorchOptionalTensor grad_hy, XPtrTorchOptionalTensor grad_cy, XPtrTorchTensor input_gates, XPtrTorchTensor hidden_gates, XPtrTorchOptionalTensor input_bias, XPtrTorchOptionalTensor hidden_bias, XPtrTorchTensor cx, XPtrTorchTensor cy) {
  auto r_out = lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(grad_hy.get(), grad_cy.get(), input_gates.get(), hidden_gates.get(), input_bias.get(), hidden_bias.get(), cx.get(), cy.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor (XPtrTorchTensor input_gates, XPtrTorchTensor hidden_gates, XPtrTorchTensor hx, XPtrTorchOptionalTensor input_bias, XPtrTorchOptionalTensor hidden_bias) {
  auto r_out = lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor(input_gates.get(), hidden_gates.get(), hx.get(), input_bias.get(), hidden_bias.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_backward_grad_hy_Tensor_workspace_Tensor_has_bias_bool (XPtrTorchTensor grad_hy, XPtrTorchTensor workspace, XPtrTorchbool has_bias) {
  auto r_out = lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool(grad_hy.get(), workspace.get(), has_bias.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_gru_cell_backward_grad_hy_Tensor_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor_input_bias_Tensor_hidden_bias_Tensor (XPtrTorchTensor grad_hy, XPtrTorchTensor input_gates, XPtrTorchTensor hidden_gates, XPtrTorchTensor hx, XPtrTorchOptionalTensor input_bias, XPtrTorchOptionalTensor hidden_bias) {
  auto r_out = lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor(grad_hy.get(), input_gates.get(), hidden_gates.get(), hx.get(), input_bias.get(), hidden_bias.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 4)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_input_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (XPtrTorchTensor input, XPtrTorchTensorList hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchbool batch_first) {
  auto r_out = lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool(input.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get(), batch_first.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_data_Tensor_batch_sizes_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (XPtrTorchTensor data, XPtrTorchTensor batch_sizes, XPtrTorchTensorList hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional) {
  auto r_out = lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool(data.get(), batch_sizes.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchbool batch_first) {
  auto r_out = lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get(), batch_first.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (XPtrTorchTensor data, XPtrTorchTensor batch_sizes, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional) {
  auto r_out = lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data.get(), batch_sizes.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchbool batch_first) {
  auto r_out = lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get(), batch_first.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (XPtrTorchTensor data, XPtrTorchTensor batch_sizes, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional) {
  auto r_out = lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data.get(), batch_sizes.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_batch_first_bool_bidirectional_bool (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional, XPtrTorchbool batch_first) {
  auto r_out = lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(input.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get(), batch_first.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (XPtrTorchTensor data, XPtrTorchTensor batch_sizes, XPtrTorchTensor hx, XPtrTorchTensorList params, XPtrTorchbool has_biases, XPtrTorchint64_t num_layers, XPtrTorchdouble dropout, XPtrTorchbool train, XPtrTorchbool bidirectional) {
  auto r_out = lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(data.get(), batch_sizes.get(), hx.get(), params.get(), has_biases.get(), num_layers.get(), dropout.get(), train.get(), bidirectional.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor (XPtrTorchTensor input, XPtrTorchTensorList hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchOptionalTensor b_ih, XPtrTorchOptionalTensor b_hh) {
  auto r_out = lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchOptionalTensor b_ih, XPtrTorchOptionalTensor b_hh) {
  auto r_out = lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchOptionalTensor b_ih, XPtrTorchOptionalTensor b_hh) {
  auto r_out = lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchOptionalTensor b_ih, XPtrTorchOptionalTensor b_hh) {
  auto r_out = lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (XPtrTorchTensor input, XPtrTorchTensorList hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchTensor b_ih, XPtrTorchTensor b_hh, XPtrTorchTensor packed_ih, XPtrTorchTensor packed_hh, XPtrTorchTensor col_offsets_ih, XPtrTorchTensor col_offsets_hh, XPtrTorchScalar scale_ih, XPtrTorchScalar scale_hh, XPtrTorchScalar zero_point_ih, XPtrTorchScalar zero_point_hh) {
  auto r_out = lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get(), packed_ih.get(), packed_hh.get(), col_offsets_ih.get(), col_offsets_hh.get(), scale_ih.get(), scale_hh.get(), zero_point_ih.get(), zero_point_hh.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchTensor b_ih, XPtrTorchTensor b_hh, XPtrTorchTensor packed_ih, XPtrTorchTensor packed_hh, XPtrTorchTensor col_offsets_ih, XPtrTorchTensor col_offsets_hh, XPtrTorchScalar scale_ih, XPtrTorchScalar scale_hh, XPtrTorchScalar zero_point_ih, XPtrTorchScalar zero_point_hh) {
  auto r_out = lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get(), packed_ih.get(), packed_hh.get(), col_offsets_ih.get(), col_offsets_hh.get(), scale_ih.get(), scale_hh.get(), zero_point_ih.get(), zero_point_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchTensor b_ih, XPtrTorchTensor b_hh, XPtrTorchTensor packed_ih, XPtrTorchTensor packed_hh, XPtrTorchTensor col_offsets_ih, XPtrTorchTensor col_offsets_hh, XPtrTorchScalar scale_ih, XPtrTorchScalar scale_hh, XPtrTorchScalar zero_point_ih, XPtrTorchScalar zero_point_hh) {
  auto r_out = lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get(), packed_ih.get(), packed_hh.get(), col_offsets_ih.get(), col_offsets_hh.get(), scale_ih.get(), scale_hh.get(), zero_point_ih.get(), zero_point_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantized_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (XPtrTorchTensor input, XPtrTorchTensor hx, XPtrTorchTensor w_ih, XPtrTorchTensor w_hh, XPtrTorchTensor b_ih, XPtrTorchTensor b_hh, XPtrTorchTensor packed_ih, XPtrTorchTensor packed_hh, XPtrTorchTensor col_offsets_ih, XPtrTorchTensor col_offsets_hh, XPtrTorchScalar scale_ih, XPtrTorchScalar scale_hh, XPtrTorchScalar zero_point_ih, XPtrTorchScalar zero_point_hh) {
  auto r_out = lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(input.get(), hx.get(), w_ih.get(), w_hh.get(), b_ih.get(), b_hh.get(), packed_ih.get(), packed_hh.get(), col_offsets_ih.get(), col_offsets_hh.get(), scale_ih.get(), scale_hh.get(), zero_point_ih.get(), zero_point_hh.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pack_padded_sequence_input_Tensor_lengths_Tensor_batch_first_bool (XPtrTorchTensor input, XPtrTorchTensor lengths, XPtrTorchbool batch_first) {
  auto r_out = lantern__pack_padded_sequence_tensor_tensor_bool(input.get(), lengths.get(), batch_first.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__pack_padded_sequence_backward_grad_Tensor_input_size_IntArrayRef_batch_sizes_Tensor_batch_first_bool (XPtrTorchTensor grad, XPtrTorchIntArrayRef input_size, XPtrTorchTensor batch_sizes, XPtrTorchbool batch_first) {
  auto r_out = lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(grad.get(), input_size.get(), batch_sizes.get(), batch_first.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pad_packed_sequence_data_Tensor_batch_sizes_Tensor_batch_first_bool_padding_value_Scalar_total_length_int64_t (XPtrTorchTensor data, XPtrTorchTensor batch_sizes, XPtrTorchbool batch_first, XPtrTorchScalar padding_value, XPtrTorchint64_t total_length) {
  auto r_out = lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt(data.get(), batch_sizes.get(), batch_first.get(), padding_value.get(), total_length.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchScalar value) {
  auto r_out = lantern_masked_fill_tensor_tensor_scalar(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor value) {
  auto r_out = lantern_masked_fill_tensor_tensor_tensor(self.get(), mask.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask, XPtrTorchTensor source) {
  auto r_out = lantern_masked_scatter_tensor_tensor_tensor(self.get(), mask.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_put_self_Tensor_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchbool accumulate) {
  auto r_out = lantern_put_tensor_tensor_tensor_bool(self.get(), index.get(), source.get(), accumulate.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern_index_add_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor_alpha_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchScalar alpha) {
  auto r_out = lantern_index_add_tensor_intt_tensor_tensor_scalar(self.get(), dim.get(), index.get(), source.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor_alpha_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor source, XPtrTorchScalar alpha) {
  auto r_out = lantern_index_add_tensor_dimname_tensor_tensor_scalar(self.get(), dim.get(), index.get(), source.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_index_fill_tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_index_fill_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_index_fill_tensor_dimname_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor value) {
  auto r_out = lantern_index_fill_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_out_tensor_tensor_intt_tensor_tensor(out.get(), self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_scatter_tensor_intt_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_scatter_out_tensor_tensor_intt_tensor_scalar(out.get(), self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src, XPtrTorchstring_view reduce) {
  auto r_out = lantern_scatter_tensor_intt_tensor_tensor_cstringview(self.get(), dim.get(), index.get(), src.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor_src_Tensor_reduce_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src, XPtrTorchstring_view reduce) {
  auto r_out = lantern_scatter_out_tensor_tensor_intt_tensor_tensor_cstringview(out.get(), self.get(), dim.get(), index.get(), src.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar_reduce_c10string_view (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value, XPtrTorchstring_view reduce) {
  auto r_out = lantern_scatter_tensor_intt_tensor_scalar_cstringview(self.get(), dim.get(), index.get(), value.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor_value_Scalar_reduce_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchScalar value, XPtrTorchstring_view reduce) {
  auto r_out = lantern_scatter_out_tensor_tensor_intt_tensor_scalar_cstringview(out.get(), self.get(), dim.get(), index.get(), value.get(), reduce.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchScalar value) {
  auto r_out = lantern_scatter_tensor_dimname_tensor_scalar(self.get(), dim.get(), index.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_add_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_add_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_add_out_tensor_tensor_intt_tensor_tensor(out.get(), self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchTensor src) {
  auto r_out = lantern_scatter_add_tensor_dimname_tensor_tensor(self.get(), dim.get(), index.get(), src.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_and_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_and_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_and_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_and_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_and_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___and___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern___and___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___and___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern___and___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_or_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_or_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_or_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_or_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_or_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___or___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern___or___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___or___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern___or___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_xor_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_xor_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_xor_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_xor_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_xor_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___xor___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern___xor___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___xor___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern___xor___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___lshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern___lshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___lshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern___lshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_left_shift_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_left_shift_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_left_shift_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_left_shift_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_left_shift_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_left_shift_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_left_shift_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_left_shift_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_left_shift_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_left_shift_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___rshift___self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern___rshift___tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace___rshift___self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern___rshift___tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_right_shift_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_right_shift_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_right_shift_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_right_shift_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_right_shift_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_right_shift_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_right_shift_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_bitwise_right_shift_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bitwise_right_shift_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_bitwise_right_shift_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(out.get(), self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (XPtrTorchTensor self, XPtrTorchTensor batch1, XPtrTorchTensor batch2, XPtrTorchScalar beta, XPtrTorchScalar alpha) {
  auto r_out = lantern_addbmm_tensor_tensor_tensor_scalar_scalar(self.get(), batch1.get(), batch2.get(), beta.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_diag_out_tensor_tensor_intt(out.get(), self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_diag_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_diag_backward_grad_Tensor_input_sizes_IntArrayRef_diagonal_int64_t (XPtrTorchTensor grad, XPtrTorchIntArrayRef input_sizes, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_diag_backward_tensor_intarrayref_intt(grad.get(), input_sizes.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cross_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_cross_out_tensor_tensor_tensor_intt(out.get(), self.get(), other.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cross_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_cross_tensor_tensor_intt(self.get(), other.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_triu_out_tensor_tensor_intt(out.get(), self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_triu_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_tril_out_tensor_tensor_intt(out.get(), self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t diagonal) {
  auto r_out = lantern_tril_tensor_intt(self.get(), diagonal.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tril_indices_row_int64_t_col_int64_t (XPtrTorchint64_t row, XPtrTorchint64_t col, XPtrTorchint64_t offset, XPtrTorchTensorOptions options) {
  auto r_out = lantern_tril_indices_intt_intt_intt_tensoroptions(row.get(), col.get(), offset.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_triu_indices_row_int64_t_col_int64_t (XPtrTorchint64_t row, XPtrTorchint64_t col, XPtrTorchint64_t offset, XPtrTorchTensorOptions options) {
  auto r_out = lantern_triu_indices_intt_intt_intt_tensoroptions(row.get(), col.get(), offset.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trace_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_trace_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_trace_backward_grad_Tensor_sizes_IntArrayRef (XPtrTorchTensor grad, XPtrTorchIntArrayRef sizes) {
  auto r_out = lantern_trace_backward_tensor_intarrayref(grad.get(), sizes.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_ne_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_ne_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ne_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ne_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ne_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_not_equal_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_not_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_not_equal_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_not_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_not_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_eq_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_eq_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_eq_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_eq_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_eq_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_ge_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_ge_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ge_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ge_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_ge_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_greater_equal_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_greater_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_greater_equal_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_greater_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_le_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_le_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_le_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_le_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_le_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_less_equal_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_less_equal_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_less_equal_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_less_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_gt_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_gt_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_gt_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gt_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_gt_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_greater_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_greater_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_greater_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_greater_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_greater_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_lt_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_lt_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_lt_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lt_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_lt_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_less_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_less_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_less_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_less_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_less_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_out_out_Tensor_self_Tensor_index_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexTensor index) {
  auto r_out = lantern_take_out_tensor_tensor_tensor(out.get(), self.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_self_Tensor_index_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor index) {
  auto r_out = lantern_take_tensor_tensor(self.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_along_dim_out_out_Tensor_self_Tensor_indices_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_take_along_dim_out_tensor_tensor_tensor_intt(out.get(), self.get(), indices.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_take_along_dim_self_Tensor_indices_Tensor (XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchoptional_index_int64_t dim) {
  auto r_out = lantern_take_along_dim_tensor_tensor_intt(self.get(), indices.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_index_select_out_tensor_tensor_intt_tensor(out.get(), self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_index_select_tensor_intt_tensor(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_index_select_out_tensor_tensor_dimname_tensor(out.get(), self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_index_select_tensor_dimname_tensor(self.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_index_select_backward_grad_Tensor_self_sizes_IntArrayRef_dim_int64_t_index_Tensor (XPtrTorchTensor grad, XPtrTorchIntArrayRef self_sizes, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index) {
  auto r_out = lantern_index_select_backward_tensor_intarrayref_intt_tensor(grad.get(), self_sizes.get(), dim.get(), index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_out_out_Tensor_self_Tensor_mask_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor mask) {
  auto r_out = lantern_masked_select_out_tensor_tensor_tensor(out.get(), self.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_self_Tensor_mask_Tensor (XPtrTorchTensor self, XPtrTorchTensor mask) {
  auto r_out = lantern_masked_select_tensor_tensor(self.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_masked_select_backward_grad_Tensor_input_Tensor_mask_Tensor (XPtrTorchTensor grad, XPtrTorchTensor input, XPtrTorchTensor mask) {
  auto r_out = lantern_masked_select_backward_tensor_tensor_tensor(grad.get(), input.get(), mask.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nonzero_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_nonzero_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nonzero_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_nonzero_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_nonzero_numpy_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_nonzero_numpy_tensor(self.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_gather_out_tensor_tensor_intt_tensor_bool(out.get(), self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_self_Tensor_dim_int64_t_index_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_gather_tensor_intt_tensor_bool(self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_backward_grad_Tensor_self_Tensor_dim_int64_t_index_Tensor_sparse_grad_bool (XPtrTorchTensor grad, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_gather_backward_tensor_tensor_intt_tensor_bool(grad.get(), self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_gather_out_tensor_tensor_dimname_tensor_bool(out.get(), self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_gather_self_Tensor_dim_Dimname_index_Tensor (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchIndexTensor index, XPtrTorchbool sparse_grad) {
  auto r_out = lantern_gather_tensor_dimname_tensor_bool(self.get(), dim.get(), index.get(), sparse_grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__gather_sparse_backward_self_Tensor_dim_int64_t_index_Tensor_grad_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor grad) {
  auto r_out = lantern__gather_sparse_backward_tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), grad.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcmul_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(out.get(), self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_addcmul_tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcdiv_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(out.get(), self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (XPtrTorchTensor self, XPtrTorchTensor tensor1, XPtrTorchTensor tensor2, XPtrTorchScalar value) {
  auto r_out = lantern_addcdiv_tensor_tensor_tensor_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cross_entropy_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index, XPtrTorchdouble label_smoothing) {
  auto r_out = lantern_cross_entropy_loss_tensor_tensor_tensor_intt_intt_double(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get(), label_smoothing.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_out_X_Tensor_qr_Tensor_self_Tensor_A_Tensor (XPtrTorchTensor X, XPtrTorchTensor qr, XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_lstsq_out_tensor_tensor_tensor_tensor(X.get(), qr.get(), self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_lstsq_tensor_tensor(self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_out_X_Tensor_M_Tensor_self_Tensor_A_Tensor (XPtrTorchTensor X, XPtrTorchTensor M, XPtrTorchTensor self, XPtrTorchTensor A, XPtrTorchbool upper, XPtrTorchbool transpose, XPtrTorchbool unitriangular) {
  auto r_out = lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool(X.get(), M.get(), self.get(), A.get(), upper.get(), transpose.get(), unitriangular.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A, XPtrTorchbool upper, XPtrTorchbool transpose, XPtrTorchbool unitriangular) {
  auto r_out = lantern_triangular_solve_tensor_tensor_bool_bool_bool(self.get(), A.get(), upper.get(), transpose.get(), unitriangular.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_out_e_Tensor_V_Tensor_self_Tensor (XPtrTorchTensor e, XPtrTorchTensor V, XPtrTorchTensor self, XPtrTorchbool eigenvectors, XPtrTorchbool upper) {
  auto r_out = lantern_symeig_out_tensor_tensor_tensor_bool_bool(e.get(), V.get(), self.get(), eigenvectors.get(), upper.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_self_Tensor (XPtrTorchTensor self, XPtrTorchbool eigenvectors, XPtrTorchbool upper) {
  auto r_out = lantern_symeig_tensor_bool_bool(self.get(), eigenvectors.get(), upper.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__symeig_helper_self_Tensor_eigenvectors_bool_upper_bool (XPtrTorchTensor self, XPtrTorchbool eigenvectors, XPtrTorchbool upper) {
  auto r_out = lantern__symeig_helper_tensor_bool_bool(self.get(), eigenvectors.get(), upper.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_out_e_Tensor_v_Tensor_self_Tensor (XPtrTorchTensor e, XPtrTorchTensor v, XPtrTorchTensor self, XPtrTorchbool eigenvectors) {
  auto r_out = lantern_eig_out_tensor_tensor_tensor_bool(e.get(), v.get(), self.get(), eigenvectors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_self_Tensor (XPtrTorchTensor self, XPtrTorchbool eigenvectors) {
  auto r_out = lantern_eig_tensor_bool(self.get(), eigenvectors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_out_U_Tensor_S_Tensor_V_Tensor_self_Tensor (XPtrTorchTensor U, XPtrTorchTensor S, XPtrTorchTensor V, XPtrTorchTensor self, XPtrTorchbool some, XPtrTorchbool compute_uv) {
  auto r_out = lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool(U.get(), S.get(), V.get(), self.get(), some.get(), compute_uv.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_self_Tensor (XPtrTorchTensor self, XPtrTorchbool some, XPtrTorchbool compute_uv) {
  auto r_out = lantern_svd_tensor_bool_bool(self.get(), some.get(), compute_uv.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__svd_helper_self_Tensor_some_bool_compute_uv_bool (XPtrTorchTensor self, XPtrTorchbool some, XPtrTorchbool compute_uv) {
  auto r_out = lantern__svd_helper_tensor_bool_bool(self.get(), some.get(), compute_uv.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_swapaxes_self_Tensor_axis0_int64_t_axis1_int64_t (XPtrTorchTensor self, XPtrTorchint64_t axis0, XPtrTorchint64_t axis1) {
  auto r_out = lantern_swapaxes_tensor_intt_intt(self.get(), axis0.get(), axis1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_swapdims_self_Tensor_dim0_int64_t_dim1_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim0, XPtrTorchindex_int64_t dim1) {
  auto r_out = lantern_swapdims_tensor_intt_intt(self.get(), dim0.get(), dim1.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_out_tensor_tensor_bool(out.get(), self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_tensor_bool(self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_solve_out_out_Tensor_self_Tensor_input2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_solve_out_tensor_tensor_tensor_bool(out.get(), self.get(), input2.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_solve_self_Tensor_input2_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_solve_tensor_tensor_bool(self.get(), input2.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cholesky_solve_helper_self_Tensor_A_Tensor_upper_bool (XPtrTorchTensor self, XPtrTorchTensor A, XPtrTorchbool upper) {
  auto r_out = lantern__cholesky_solve_helper_tensor_tensor_bool(self.get(), A.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_solve_tensor_tensor(self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_out_solution_Tensor_lu_Tensor_self_Tensor_A_Tensor (XPtrTorchTensor solution, XPtrTorchTensor lu, XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern_solve_out_tensor_tensor_tensor_tensor(solution.get(), lu.get(), self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__solve_helper_self_Tensor_A_Tensor (XPtrTorchTensor self, XPtrTorchTensor A) {
  auto r_out = lantern__solve_helper_tensor_tensor(self.get(), A.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_inverse_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_inverse_tensor_bool(self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_cholesky_inverse_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_cholesky_inverse_out_tensor_tensor_bool(out.get(), self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_out_Q_Tensor_R_Tensor_self_Tensor (XPtrTorchTensor Q, XPtrTorchTensor R, XPtrTorchTensor self, XPtrTorchbool some) {
  auto r_out = lantern_qr_out_tensor_tensor_tensor_bool(Q.get(), R.get(), self.get(), some.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_self_Tensor (XPtrTorchTensor self, XPtrTorchbool some) {
  auto r_out = lantern_qr_tensor_bool(self.get(), some.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_out_a_Tensor_tau_Tensor_self_Tensor (XPtrTorchTensor a, XPtrTorchTensor tau, XPtrTorchTensor self) {
  auto r_out = lantern_geqrf_out_tensor_tensor_tensor(a.get(), tau.get(), self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_geqrf_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_orgqr_self_Tensor_input2_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2) {
  auto r_out = lantern_orgqr_tensor_tensor(self.get(), input2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_orgqr_out_out_Tensor_self_Tensor_input2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor input2) {
  auto r_out = lantern_orgqr_out_tensor_tensor_tensor(out.get(), self.get(), input2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ormqr_out_out_Tensor_self_Tensor_input2_Tensor_input3_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchTensor input3, XPtrTorchbool left, XPtrTorchbool transpose) {
  auto r_out = lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool(out.get(), self.get(), input2.get(), input3.get(), left.get(), transpose.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ormqr_self_Tensor_input2_Tensor_input3_Tensor (XPtrTorchTensor self, XPtrTorchTensor input2, XPtrTorchTensor input3, XPtrTorchbool left, XPtrTorchbool transpose) {
  auto r_out = lantern_ormqr_tensor_tensor_tensor_bool_bool(self.get(), input2.get(), input3.get(), left.get(), transpose.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__lu_with_info_self_Tensor (XPtrTorchTensor self, XPtrTorchbool pivot, XPtrTorchbool check_errors) {
  auto r_out = lantern__lu_with_info_tensor_bool_bool(self.get(), pivot.get(), check_errors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lu_solve_out_out_Tensor_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor LU_data, XPtrTorchTensor LU_pivots) {
  auto r_out = lantern_lu_solve_out_tensor_tensor_tensor_tensor(out.get(), self.get(), LU_data.get(), LU_pivots.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (XPtrTorchTensor self, XPtrTorchTensor LU_data, XPtrTorchTensor LU_pivots) {
  auto r_out = lantern_lu_solve_tensor_tensor_tensor(self.get(), LU_data.get(), LU_pivots.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lu_unpack_LU_data_Tensor_LU_pivots_Tensor (XPtrTorchTensor LU_data, XPtrTorchTensor LU_pivots, XPtrTorchbool unpack_data, XPtrTorchbool unpack_pivots) {
  auto r_out = lantern_lu_unpack_tensor_tensor_bool_bool(LU_data.get(), LU_pivots.get(), unpack_data.get(), unpack_pivots.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lu_unpack_out_P_Tensor_L_Tensor_U_Tensor_LU_data_Tensor_LU_pivots_Tensor (XPtrTorchTensor P, XPtrTorchTensor L, XPtrTorchTensor U, XPtrTorchTensor LU_data, XPtrTorchTensor LU_pivots, XPtrTorchbool unpack_data, XPtrTorchbool unpack_pivots) {
  auto r_out = lantern_lu_unpack_out_tensor_tensor_tensor_tensor_tensor_bool_bool(P.get(), L.get(), U.get(), LU_data.get(), LU_pivots.get(), unpack_data.get(), unpack_pivots.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multinomial_out_out_Tensor_self_Tensor_num_samples_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t num_samples, XPtrTorchbool replacement, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_multinomial_out_tensor_tensor_intt_bool_generator(out.get(), self.get(), num_samples.get(), replacement.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multinomial_self_Tensor_num_samples_int64_t (XPtrTorchTensor self, XPtrTorchint64_t num_samples, XPtrTorchbool replacement, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_multinomial_tensor_intt_bool_generator(self.get(), num_samples.get(), replacement.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lgamma_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_lgamma_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lgamma_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_lgamma_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_digamma_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_digamma_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_digamma_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_digamma_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_polygamma_out_out_Tensor_n_int64_t_self_Tensor (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchTensor self) {
  auto r_out = lantern_polygamma_out_tensor_intt_tensor(out.get(), n.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfinv_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_erfinv_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_erfinv_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_erfinv_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_i0_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_i0__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_i0_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_i0_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sign_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_sign_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sign_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_sign_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_signbit_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_signbit_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_signbit_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_signbit_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_dist_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar p) {
  auto r_out = lantern_dist_tensor_tensor_scalar(self.get(), other.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan2_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_atan2_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_atan2_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_atan2_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchScalar weight) {
  auto r_out = lantern_lerp_out_tensor_tensor_tensor_scalar(out.get(), self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchTensor weight) {
  auto r_out = lantern_lerp_out_tensor_tensor_tensor_tensor(out.get(), self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Scalar (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchScalar weight) {
  auto r_out = lantern_lerp_tensor_tensor_scalar(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Tensor (XPtrTorchTensor self, XPtrTorchTensor end, XPtrTorchTensor weight) {
  auto r_out = lantern_lerp_tensor_tensor_tensor(self.get(), end.get(), weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_histc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchScalar min, XPtrTorchScalar max) {
  auto r_out = lantern_histc_out_tensor_tensor_intt_scalar_scalar(out.get(), self.get(), bins.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_histc_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchScalar min, XPtrTorchScalar max) {
  auto r_out = lantern_histc_tensor_intt_scalar_scalar(self.get(), bins.get(), min.get(), max.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_histogram_out_hist_Tensor_bin_edges_Tensor_self_Tensor_bins_Tensor (XPtrTorchTensor hist, XPtrTorchTensor bin_edges, XPtrTorchTensor self, XPtrTorchTensor bins, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_histogram_out_tensor_tensor_tensor_tensor_tensor_bool(hist.get(), bin_edges.get(), self.get(), bins.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_histogram_self_Tensor_bins_Tensor (XPtrTorchTensor self, XPtrTorchTensor bins, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_histogram_tensor_tensor_tensor_bool(self.get(), bins.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_histogram_out_hist_Tensor_bin_edges_Tensor_self_Tensor_bins_int64_t (XPtrTorchTensor hist, XPtrTorchTensor bin_edges, XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchOptionalDoubleArrayRef range, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_histogram_out_tensor_tensor_tensor_intt_arrayrefdouble_tensor_bool(hist.get(), bin_edges.get(), self.get(), bins.get(), range.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_histogram_self_Tensor_bins_int64_t (XPtrTorchTensor self, XPtrTorchint64_t bins, XPtrTorchOptionalDoubleArrayRef range, XPtrTorchOptionalTensor weight, XPtrTorchbool density) {
  auto r_out = lantern_histogram_tensor_intt_arrayrefdouble_tensor_bool(self.get(), bins.get(), range.get(), weight.get(), density.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_fmod_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_fmod_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmod_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmod_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmod_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hypot_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_hypot_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hypot_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_hypot_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_igamma_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_igamma_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_igamma_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_igamma_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_igammac_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_igammac_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_igammac_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_igammac_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nextafter_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_nextafter_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nextafter_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_nextafter_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_remainder_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_remainder_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_remainder_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_remainder_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_remainder_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_remainder_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_min_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmin_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmin_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmin_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmin_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_max_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmax_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmax_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fmax_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_fmax_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_maximum_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_maximum_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_maximum_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_maximum_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_max_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_out_out_Tensor_other_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_max_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_minimum_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_minimum_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_minimum_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_minimum_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_out_out_Tensor_other_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_min_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_min_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_min_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_quantile_out_tensor_tensor_double_intt_bool(out.get(), self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_quantile_tensor_double_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_quantile_out_tensor_tensor_tensor_intt_bool(out.get(), self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_quantile_tensor_tensor_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_double_intt_bool(out.get(), self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_double_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanquantile_tensor_double_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_tensor_intt_bool(out.get(), self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_nanquantile_tensor_tensor_intt_bool(self.get(), q.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_quantile_out_tensor_tensor_double_intt_bool_cstringview(out.get(), self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_quantile_tensor_double_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_out_out_Tensor_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_quantile_out_tensor_tensor_tensor_intt_bool_cstringview(out.get(), self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_quantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_quantile_tensor_tensor_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_double_intt_bool_cstringview(out.get(), self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_double_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchdouble q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_nanquantile_tensor_double_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_out_out_Tensor_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_nanquantile_out_tensor_tensor_tensor_intt_bool_cstringview(out.get(), self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nanquantile_self_Tensor_q_Tensor_dim_int64_t_keepdim_bool_interpolation_c10string_view (XPtrTorchTensor self, XPtrTorchTensor q, XPtrTorchoptional_index_int64_t dim, XPtrTorchbool keepdim, XPtrTorchstring_view interpolation) {
  auto r_out = lantern_nanquantile_tensor_tensor_intt_bool_cstringview(self.get(), q.get(), dim.get(), keepdim.get(), interpolation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_intt_bool(values.get(), indices.get(), self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_stable_bool_dim_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_bool_intt_bool(values.get(), indices.get(), self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_tensor_intt_bool(self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_int64_t_stable_bool (XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_tensor_bool_intt_bool(self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_dimname_bool(values.get(), indices.get(), self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_stable_bool_dim_Dimname (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_out_tensor_tensor_tensor_bool_dimname_bool(values.get(), indices.get(), self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_tensor_dimname_bool(self.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_Dimname_stable_bool (XPtrTorchTensor self, XPtrTorchoptional_bool stable, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_sort_tensor_bool_dimname_bool(self.get(), stable.get(), dim.get(), descending.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_msort_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_msort_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_msort_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_msort_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argsort_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchbool descending) {
  auto r_out = lantern_argsort_tensor_intt_bool(self.get(), dim.get(), descending.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_argsort_self_Tensor_dim_Dimname (XPtrTorchTensor self, XPtrTorchDimname dim, XPtrTorchbool descending) {
  auto r_out = lantern_argsort_tensor_dimname_bool(self.get(), dim.get(), descending.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t (XPtrTorchTensor values, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool largest, XPtrTorchbool sorted) {
  auto r_out = lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool(values.get(), indices.get(), self.get(), k.get(), dim.get(), largest.get(), sorted.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_self_Tensor_k_int64_t (XPtrTorchTensor self, XPtrTorchint64_t k, XPtrTorchindex_int64_t dim, XPtrTorchbool largest, XPtrTorchbool sorted) {
  auto r_out = lantern_topk_tensor_intt_intt_bool_bool(self.get(), k.get(), dim.get(), largest.get(), sorted.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_all_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_all_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_all_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_any_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_any_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_any_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_renorm_out_out_Tensor_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar p, XPtrTorchindex_int64_t dim, XPtrTorchScalar maxnorm) {
  auto r_out = lantern_renorm_out_tensor_tensor_scalar_intt_scalar(out.get(), self.get(), p.get(), dim.get(), maxnorm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (XPtrTorchTensor self, XPtrTorchScalar p, XPtrTorchindex_int64_t dim, XPtrTorchScalar maxnorm) {
  auto r_out = lantern_renorm_tensor_scalar_intt_scalar(self.get(), p.get(), dim.get(), maxnorm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_unfold_backward_grad_in_Tensor_input_sizes_IntArrayRef_dim_int64_t_size_int64_t_step_int64_t (XPtrTorchTensor grad_in, XPtrTorchIntArrayRef input_sizes, XPtrTorchindex_int64_t dim, XPtrTorchint64_t size, XPtrTorchint64_t step) {
  auto r_out = lantern_unfold_backward_tensor_intarrayref_intt_intt_intt(grad_in.get(), input_sizes.get(), dim.get(), size.get(), step.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchbool cpp_torch_namespace_equal_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_equal_tensor_tensor(self.get(), other.get());
return XPtrTorchbool(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_pow_out_tensor_tensor_tensor(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_pow_tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Scalar_exponent_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor exponent) {
  auto r_out = lantern_pow_out_tensor_scalar_tensor(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Scalar_exponent_Tensor (XPtrTorchScalar self, XPtrTorchTensor exponent) {
  auto r_out = lantern_pow_scalar_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_pow_out_tensor_tensor_scalar(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pow_self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_pow_tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_out_out_Tensor_self_Tensor_exponent_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_float_power_out_tensor_tensor_tensor(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_self_Tensor_exponent_Tensor (XPtrTorchTensor self, XPtrTorchTensor exponent) {
  auto r_out = lantern_float_power_tensor_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_out_out_Tensor_self_Scalar_exponent_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor exponent) {
  auto r_out = lantern_float_power_out_tensor_scalar_tensor(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_self_Scalar_exponent_Tensor (XPtrTorchScalar self, XPtrTorchTensor exponent) {
  auto r_out = lantern_float_power_scalar_tensor(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_out_out_Tensor_self_Tensor_exponent_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_float_power_out_tensor_tensor_scalar(out.get(), self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_float_power_self_Tensor_exponent_Scalar (XPtrTorchTensor self, XPtrTorchScalar exponent) {
  auto r_out = lantern_float_power_tensor_scalar(self.get(), exponent.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor_std_double (XPtrTorchTensor out, XPtrTorchTensor mean, XPtrTorchdouble std, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_normal_out_tensor_tensor_double_generator(out.get(), mean.get(), std.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_Tensor (XPtrTorchTensor out, XPtrTorchdouble mean, XPtrTorchTensor std, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_normal_out_tensor_double_tensor_generator(out.get(), mean.get(), std.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor_std_Tensor (XPtrTorchTensor out, XPtrTorchTensor mean, XPtrTorchTensor std, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_normal_out_tensor_tensor_tensor_generator(out.get(), mean.get(), std.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_double_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchdouble mean, XPtrTorchdouble std, XPtrTorchIntArrayRef size, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_normal_out_tensor_double_double_intarrayref_generator(out.get(), mean.get(), std.get(), size.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_alias_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_alias_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchIndexTensor index, XPtrTorchTensor source) {
  auto r_out = lantern__index_copy__tensor_intt_tensor_tensor(self.get(), dim.get(), index.get(), source.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__amp_foreach_non_finite_check_and_unscale__self_TensorList_found_inf_Tensor_inv_scale_Tensor (XPtrTorchTensorList self, XPtrTorchTensor found_inf, XPtrTorchTensor inv_scale) {
  lantern__amp_foreach_non_finite_check_and_unscale__tensorlist_tensor_tensor(self.get(), found_inf.get(), inv_scale.get());
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__amp_update_scale__self_Tensor_growth_tracker_Tensor_found_inf_Tensor_scale_growth_factor_double_scale_backoff_factor_double_growth_interval_int64_t (XPtrTorchTensor self, XPtrTorchTensor growth_tracker, XPtrTorchTensor found_inf, XPtrTorchdouble scale_growth_factor, XPtrTorchdouble scale_backoff_factor, XPtrTorchint64_t growth_interval) {
  auto r_out = lantern__amp_update_scale__tensor_tensor_tensor_double_double_intt(self.get(), growth_tracker.get(), found_inf.get(), scale_growth_factor.get(), scale_backoff_factor.get(), growth_interval.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cat_tensors_TensorList (XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__cat_tensorlist_intt(tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__cat_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern__cat_out_tensor_tensorlist_intt(out.get(), tensors.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_add_tensors_TensorList_scalar_Scalar (XPtrTorchTensorList tensors, XPtrTorchScalar scalar) {
  auto r_out = lantern__foreach_add_tensorlist_scalar(tensors.get(), scalar.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add__self_TensorList_scalar_Scalar (XPtrTorchTensorList self, XPtrTorchScalar scalar) {
  lantern__foreach_add__tensorlist_scalar(self.get(), scalar.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sub_tensors_TensorList_scalar_Scalar (XPtrTorchTensorList tensors, XPtrTorchScalar scalar) {
  auto r_out = lantern__foreach_sub_tensorlist_scalar(tensors.get(), scalar.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub__self_TensorList_scalar_Scalar (XPtrTorchTensorList self, XPtrTorchScalar scalar) {
  lantern__foreach_sub__tensorlist_scalar(self.get(), scalar.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_mul_tensors_TensorList_scalar_Scalar (XPtrTorchTensorList tensors, XPtrTorchScalar scalar) {
  auto r_out = lantern__foreach_mul_tensorlist_scalar(tensors.get(), scalar.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul__self_TensorList_scalar_Scalar (XPtrTorchTensorList self, XPtrTorchScalar scalar) {
  lantern__foreach_mul__tensorlist_scalar(self.get(), scalar.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_div_tensors_TensorList_scalar_Scalar (XPtrTorchTensorList tensors, XPtrTorchScalar scalar) {
  auto r_out = lantern__foreach_div_tensorlist_scalar(tensors.get(), scalar.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div__self_TensorList_scalar_Scalar (XPtrTorchTensorList self, XPtrTorchScalar scalar) {
  lantern__foreach_div__tensorlist_scalar(self.get(), scalar.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_add_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2, XPtrTorchScalar alpha) {
  auto r_out = lantern__foreach_add_tensorlist_tensorlist_scalar(tensors1.get(), tensors2.get(), alpha.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add__self_TensorList_other_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList other, XPtrTorchScalar alpha) {
  lantern__foreach_add__tensorlist_tensorlist_scalar(self.get(), other.get(), alpha.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sub_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2, XPtrTorchScalar alpha) {
  auto r_out = lantern__foreach_sub_tensorlist_tensorlist_scalar(tensors1.get(), tensors2.get(), alpha.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub__self_TensorList_other_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList other, XPtrTorchScalar alpha) {
  lantern__foreach_sub__tensorlist_tensorlist_scalar(self.get(), other.get(), alpha.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_mul_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2) {
  auto r_out = lantern__foreach_mul_tensorlist_tensorlist(tensors1.get(), tensors2.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul__self_TensorList_other_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList other) {
  lantern__foreach_mul__tensorlist_tensorlist(self.get(), other.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_div_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2) {
  auto r_out = lantern__foreach_div_tensorlist_tensorlist(tensors1.get(), tensors2.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div__self_TensorList_other_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList other) {
  lantern__foreach_div__tensorlist_tensorlist(self.get(), other.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_add_tensors_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList tensors, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_add_tensorlist_arrayrefscalar(tensors.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_add__self_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_add__tensorlist_arrayrefscalar(self.get(), scalars.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sub_tensors_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList tensors, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_sub_tensorlist_arrayrefscalar(tensors.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sub__self_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_sub__tensorlist_arrayrefscalar(self.get(), scalars.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_div_tensors_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList tensors, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_div_tensorlist_arrayrefscalar(tensors.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_div__self_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_div__tensorlist_arrayrefscalar(self.get(), scalars.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_mul_tensors_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList tensors, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_mul_tensorlist_arrayrefscalar(tensors.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_mul__self_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_mul__tensorlist_arrayrefscalar(self.get(), scalars.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_exp_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_exp_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_zero__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_zero__tensorlist(self.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_exp__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_exp__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sqrt_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_sqrt_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sqrt__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_sqrt__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_abs_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_abs_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_abs__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_abs__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_acos_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_acos_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_acos__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_acos__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_asin_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_asin_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_asin__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_asin__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_atan_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_atan_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_atan__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_atan__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_ceil_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_ceil_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_ceil__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_ceil__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_cos_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_cos_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_cos__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_cos__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_cosh_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_cosh_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_cosh__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_cosh__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_erf_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_erf_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_erf__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_erf__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_erfc_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_erfc_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_erfc__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_erfc__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_expm1_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_expm1_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_expm1__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_expm1__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_floor_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_floor_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_floor__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_floor__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_log_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_log_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_log__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_log__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_log10_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_log10_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_log10__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_log10__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_log1p_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_log1p_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_log1p__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_log1p__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_log2_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_log2_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_log2__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_log2__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_neg_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_neg_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_neg__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_neg__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_tan_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_tan_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_tan__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_tan__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_tanh_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_tanh_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_tanh__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_tanh__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sin_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_sin_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sin__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_sin__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sinh_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_sinh_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sinh__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_sinh__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_round_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_round_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_round__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_round__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_lgamma_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_lgamma_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_lgamma__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_lgamma__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_frac_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_frac_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_frac__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_frac__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_reciprocal_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_reciprocal_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_reciprocal__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_reciprocal__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_sigmoid_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_sigmoid_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_sigmoid__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_sigmoid__tensorlist(self.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_trunc_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern__foreach_trunc_tensorlist(tensors.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_trunc__self_TensorList (XPtrTorchTensorList self) {
  lantern__foreach_trunc__tensorlist(self.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcdiv__self_TensorList_tensor1_TensorList_tensor2_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchScalar value) {
  lantern__foreach_addcdiv__tensorlist_tensorlist_tensorlist_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcmul__self_TensorList_tensor1_TensorList_tensor2_TensorList (XPtrTorchTensorList self, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchScalar value) {
  lantern__foreach_addcmul__tensorlist_tensorlist_tensorlist_scalar(self.get(), tensor1.get(), tensor2.get(), value.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcdiv__self_TensorList_tensor1_TensorList_tensor2_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_addcdiv__tensorlist_tensorlist_tensorlist_arrayrefscalar(self.get(), tensor1.get(), tensor2.get(), scalars.get());
}

// [[Rcpp::export]]
void cpp_torch_namespace__foreach_addcmul__self_TensorList_tensor1_TensorList_tensor2_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList self, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchvector_Scalar scalars) {
  lantern__foreach_addcmul__tensorlist_tensorlist_tensorlist_arrayrefscalar(self.get(), tensor1.get(), tensor2.get(), scalars.get());
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_addcdiv_input_TensorList_tensor1_TensorList_tensor2_TensorList (XPtrTorchTensorList input, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchScalar value) {
  auto r_out = lantern__foreach_addcdiv_tensorlist_tensorlist_tensorlist_scalar(input.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_addcmul_input_TensorList_tensor1_TensorList_tensor2_TensorList (XPtrTorchTensorList input, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchScalar value) {
  auto r_out = lantern__foreach_addcmul_tensorlist_tensorlist_tensorlist_scalar(input.get(), tensor1.get(), tensor2.get(), value.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_addcdiv_input_TensorList_tensor1_TensorList_tensor2_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList input, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_addcdiv_tensorlist_tensorlist_tensorlist_arrayrefscalar(input.get(), tensor1.get(), tensor2.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_addcmul_input_TensorList_tensor1_TensorList_tensor2_TensorList_scalars_ArrayRefScalar (XPtrTorchTensorList input, XPtrTorchTensorList tensor1, XPtrTorchTensorList tensor2, XPtrTorchvector_Scalar scalars) {
  auto r_out = lantern__foreach_addcmul_tensorlist_tensorlist_tensorlist_arrayrefscalar(input.get(), tensor1.get(), tensor2.get(), scalars.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_maximum_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2) {
  auto r_out = lantern__foreach_maximum_tensorlist_tensorlist(tensors1.get(), tensors2.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace__foreach_minimum_tensors1_TensorList_tensors2_TensorList (XPtrTorchTensorList tensors1, XPtrTorchTensorList tensors2) {
  auto r_out = lantern__foreach_minimum_tensorlist_tensorlist(tensors1.get(), tensors2.get());
return XPtrTorchTensorList(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_self_Tensor_boundaries_Tensor (XPtrTorchTensor self, XPtrTorchTensor boundaries, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_bucketize_tensor_tensor_bool_bool(self.get(), boundaries.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_out_out_Tensor_self_Tensor_boundaries_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor boundaries, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_bucketize_out_tensor_tensor_tensor_bool_bool(out.get(), self.get(), boundaries.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_bucketize_self_Scalar_boundaries_Tensor (XPtrTorchScalar self, XPtrTorchTensor boundaries, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_bucketize_scalar_tensor_bool_bool(self.get(), boundaries.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_sorted_sequence_Tensor_self_Tensor (XPtrTorchTensor sorted_sequence, XPtrTorchTensor self, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_searchsorted_tensor_tensor_bool_bool(sorted_sequence.get(), self.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_out_out_Tensor_sorted_sequence_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor sorted_sequence, XPtrTorchTensor self, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_searchsorted_out_tensor_tensor_tensor_bool_bool(out.get(), sorted_sequence.get(), self.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_searchsorted_sorted_sequence_Tensor_self_Scalar (XPtrTorchTensor sorted_sequence, XPtrTorchScalar self, XPtrTorchbool out_int32, XPtrTorchbool right) {
  auto r_out = lantern_searchsorted_tensor_scalar_bool_bool(sorted_sequence.get(), self.get(), out_int32.get(), right.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convert_indices_from_coo_to_csr_self_Tensor_size_int64_t (XPtrTorchTensor self, XPtrTorchint64_t size, XPtrTorchbool out_int32) {
  auto r_out = lantern__convert_indices_from_coo_to_csr_tensor_intt_bool(self.get(), size.get(), out_int32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__convert_indices_from_coo_to_csr_out_out_Tensor_self_Tensor_size_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t size, XPtrTorchbool out_int32) {
  auto r_out = lantern__convert_indices_from_coo_to_csr_out_tensor_tensor_intt_bool(out.get(), self.get(), size.get(), out_int32.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_mse_loss_out_tensor_tensor_tensor_intt(out.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_mse_loss_tensor_tensor_intt(self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mse_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_mse_loss_backward_tensor_tensor_tensor_intt(grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_l1_loss_out_tensor_tensor_tensor_intt(out.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_l1_loss_tensor_tensor_intt(self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_l1_loss_backward_tensor_tensor_tensor_intt(grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchScalar p, XPtrTorchScalar margin, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt(out.get(), self.get(), target.get(), p.get(), margin.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchScalar p, XPtrTorchScalar margin, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt(self.get(), target.get(), p.get(), margin.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchScalar p, XPtrTorchScalar margin, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt(grad_input.get(), grad_output.get(), self.get(), target.get(), p.get(), margin.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multi_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchScalar p, XPtrTorchScalar margin, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt(grad_output.get(), self.get(), target.get(), p.get(), margin.get(), weight.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(out.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multilabel_margin_loss_tensor_tensor_intt(self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_out_output_Tensor_is_target_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor output, XPtrTorchTensor is_target, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt(output.get(), is_target.get(), self.get(), target.get(), reduction.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_multilabel_margin_loss_forward_tensor_tensor_intt(self.get(), target.get(), reduction.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchTensor is_target) {
  auto r_out = lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get(), is_target.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_multilabel_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchTensor is_target) {
  auto r_out = lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor(grad_output.get(), self.get(), target.get(), reduction.get(), is_target.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(out.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_nd_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss_nd_tensor_tensor_tensor_intt_intt(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss_tensor_tensor_tensor_intt_intt(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (XPtrTorchTensor output, XPtrTorchTensor total_weight, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(output.get(), total_weight.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index, XPtrTorchTensor total_weight) {
  auto r_out = lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_input.get(), grad_output.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get(), total_weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index, XPtrTorchTensor total_weight) {
  auto r_out = lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_output.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get(), total_weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(out.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (XPtrTorchTensor output, XPtrTorchTensor total_weight, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(output.get(), total_weight.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index) {
  auto r_out = lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt(self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index, XPtrTorchTensor total_weight) {
  auto r_out = lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_input.get(), grad_output.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get(), total_weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_nll_loss2d_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchOptionalTensor weight, XPtrTorchint64_t reduction, XPtrTorchint64_t ignore_index, XPtrTorchTensor total_weight) {
  auto r_out = lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(grad_output.get(), self.get(), target.get(), weight.get(), reduction.get(), ignore_index.get(), total_weight.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble beta) {
  auto r_out = lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt_double(out.get(), self.get(), target.get(), reduction.get(), beta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble beta) {
  auto r_out = lantern_smooth_l1_loss_tensor_tensor_intt_double(self.get(), target.get(), reduction.get(), beta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_beta_double (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble beta) {
  auto r_out = lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get(), beta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_smooth_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_beta_double (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble beta) {
  auto r_out = lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt_double(grad_output.get(), self.get(), target.get(), reduction.get(), beta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_huber_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble delta) {
  auto r_out = lantern_huber_loss_out_tensor_tensor_tensor_intt_double(out.get(), self.get(), target.get(), reduction.get(), delta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_huber_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble delta) {
  auto r_out = lantern_huber_loss_tensor_tensor_intt_double(self.get(), target.get(), reduction.get(), delta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_huber_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_delta_double (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble delta) {
  auto r_out = lantern_huber_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get(), delta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_huber_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_delta_double (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction, XPtrTorchdouble delta) {
  auto r_out = lantern_huber_loss_backward_tensor_tensor_tensor_intt_double(grad_output.get(), self.get(), target.get(), reduction.get(), delta.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_soft_margin_loss_out_tensor_tensor_tensor_intt(out.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_self_Tensor_target_Tensor (XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_soft_margin_loss_tensor_tensor_intt(self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt(grad_input.get(), grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_soft_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor target, XPtrTorchint64_t reduction) {
  auto r_out = lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt(grad_output.get(), self.get(), target.get(), reduction.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar alpha, XPtrTorchScalar scale, XPtrTorchScalar input_scale) {
  auto r_out = lantern_elu_out_tensor_tensor_scalar_scalar_scalar(out.get(), self.get(), alpha.get(), scale.get(), input_scale.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar alpha, XPtrTorchScalar scale, XPtrTorchScalar input_scale) {
  auto r_out = lantern_elu_tensor_scalar_scalar_scalar(self.get(), alpha.get(), scale.get(), input_scale.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_backward_out_grad_input_Tensor_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_is_result_bool_self_or_result_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchScalar alpha, XPtrTorchScalar scale, XPtrTorchScalar input_scale, XPtrTorchbool is_result, XPtrTorchTensor self_or_result) {
  auto r_out = lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_bool_tensor(grad_input.get(), grad_output.get(), alpha.get(), scale.get(), input_scale.get(), is_result.get(), self_or_result.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu_backward_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_is_result_bool_self_or_result_Tensor (XPtrTorchTensor grad_output, XPtrTorchScalar alpha, XPtrTorchScalar scale, XPtrTorchScalar input_scale, XPtrTorchbool is_result, XPtrTorchTensor self_or_result) {
  auto r_out = lantern_elu_backward_tensor_scalar_scalar_scalar_bool_tensor(grad_output.get(), alpha.get(), scale.get(), input_scale.get(), is_result.get(), self_or_result.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_elu__self_Tensor (XPtrTorchTensor self, XPtrTorchScalar alpha, XPtrTorchScalar scale, XPtrTorchScalar input_scale) {
  auto r_out = lantern_elu__tensor_scalar_scalar_scalar(self.get(), alpha.get(), scale.get(), input_scale.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_glu_out_tensor_tensor_intt(out.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_self_Tensor (XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_glu_tensor_intt(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_glu_backward_out_tensor_tensor_tensor_intt(grad_input.get(), grad_output.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_glu_backward_grad_output_Tensor_self_Tensor_dim_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchindex_int64_t dim) {
  auto r_out = lantern_glu_backward_tensor_tensor_intt(grad_output.get(), self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_hardsigmoid_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_hardsigmoid_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_hardsigmoid__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_hardsigmoid_backward_out_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardsigmoid_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_hardsigmoid_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar min_val, XPtrTorchScalar max_val) {
  auto r_out = lantern_hardtanh_out_tensor_tensor_scalar_scalar(out.get(), self.get(), min_val.get(), max_val.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar min_val, XPtrTorchScalar max_val) {
  auto r_out = lantern_hardtanh_tensor_scalar_scalar(self.get(), min_val.get(), max_val.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar min_val, XPtrTorchScalar max_val) {
  auto r_out = lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar(grad_input.get(), grad_output.get(), self.get(), min_val.get(), max_val.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh_backward_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar min_val, XPtrTorchScalar max_val) {
  auto r_out = lantern_hardtanh_backward_tensor_tensor_scalar_scalar(grad_output.get(), self.get(), min_val.get(), max_val.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardtanh__self_Tensor (XPtrTorchTensor self, XPtrTorchScalar min_val, XPtrTorchScalar max_val) {
  auto r_out = lantern_hardtanh__tensor_scalar_scalar(self.get(), min_val.get(), max_val.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_hardswish_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_hardswish_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish__self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_hardswish__tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_hardswish_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_hardswish_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar negative_slope) {
  auto r_out = lantern_leaky_relu_out_tensor_tensor_scalar(out.get(), self.get(), negative_slope.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar negative_slope) {
  auto r_out = lantern_leaky_relu_tensor_scalar(self.get(), negative_slope.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_negative_slope_Scalar_self_is_result_bool (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar negative_slope, XPtrTorchbool self_is_result) {
  auto r_out = lantern_leaky_relu_backward_out_tensor_tensor_tensor_scalar_bool(grad_input.get(), grad_output.get(), self.get(), negative_slope.get(), self_is_result.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu_backward_grad_output_Tensor_self_Tensor_negative_slope_Scalar_self_is_result_bool (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar negative_slope, XPtrTorchbool self_is_result) {
  auto r_out = lantern_leaky_relu_backward_tensor_tensor_scalar_bool(grad_output.get(), self.get(), negative_slope.get(), self_is_result.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_leaky_relu__self_Tensor (XPtrTorchTensor self, XPtrTorchScalar negative_slope) {
  auto r_out = lantern_leaky_relu__tensor_scalar(self.get(), negative_slope.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_log_sigmoid_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log_sigmoid_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_out_output_Tensor_buffer_Tensor_self_Tensor (XPtrTorchTensor output, XPtrTorchTensor buffer, XPtrTorchTensor self) {
  auto r_out = lantern_log_sigmoid_forward_out_tensor_tensor_tensor(output.get(), buffer.get(), self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_log_sigmoid_forward_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_buffer_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor buffer) {
  auto r_out = lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get(), buffer.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_log_sigmoid_backward_grad_output_Tensor_self_Tensor_buffer_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor buffer) {
  auto r_out = lantern_log_sigmoid_backward_tensor_tensor_tensor(grad_output.get(), self.get(), buffer.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_out_out_Tensor_self_Tensor_noise_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor noise, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator(out.get(), self.get(), noise.get(), lower.get(), upper.get(), training.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_self_Tensor_noise_Tensor (XPtrTorchTensor self, XPtrTorchTensor noise, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator(self.get(), noise.get(), lower.get(), upper.get(), training.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise_backward_grad_output_Tensor_self_Tensor_noise_Tensor_lower_Scalar_upper_Scalar_training_bool_self_is_result_bool (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor noise, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchbool self_is_result) {
  auto r_out = lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool_bool(grad_output.get(), self.get(), noise.get(), lower.get(), upper.get(), training.get(), self_is_result.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_rrelu_with_noise__self_Tensor_noise_Tensor (XPtrTorchTensor self, XPtrTorchTensor noise, XPtrTorchScalar lower, XPtrTorchScalar upper, XPtrTorchbool training, XPtrTorchOptionalGenerator generator) {
  auto r_out = lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator(self.get(), noise.get(), lower.get(), upper.get(), training.get(), generator.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar beta, XPtrTorchScalar threshold) {
  auto r_out = lantern_softplus_out_tensor_tensor_scalar_scalar(out.get(), self.get(), beta.get(), threshold.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar beta, XPtrTorchScalar threshold) {
  auto r_out = lantern_softplus_tensor_scalar_scalar(self.get(), beta.get(), threshold.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar beta, XPtrTorchScalar threshold, XPtrTorchTensor output) {
  auto r_out = lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor(grad_input.get(), grad_output.get(), self.get(), beta.get(), threshold.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softplus_backward_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar beta, XPtrTorchScalar threshold, XPtrTorchTensor output) {
  auto r_out = lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor(grad_output.get(), self.get(), beta.get(), threshold.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_softshrink_out_tensor_tensor_scalar(out.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_softshrink_tensor_scalar(self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_lambd_Scalar (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_softshrink_backward_out_tensor_tensor_tensor_scalar(grad_input.get(), grad_output.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_softshrink_backward_grad_output_Tensor_self_Tensor_lambd_Scalar (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchScalar lambd) {
  auto r_out = lantern_softshrink_backward_tensor_tensor_scalar(grad_output.get(), self.get(), lambd.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref(out.get(), self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_avg_pool2d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_mkldnn_adaptive_avg_pool2d_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_mkldnn_adaptive_avg_pool2d_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern__adaptive_avg_pool2d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool2d_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern__adaptive_avg_pool2d_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref(out.get(), self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_avg_pool3d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool3d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern__adaptive_avg_pool3d_tensor_intarrayref(self.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__adaptive_avg_pool3d_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self) {
  auto r_out = lantern__adaptive_avg_pool3d_backward_tensor_tensor(grad_output.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref(out.get(), indices.get(), self.get(), output_size.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_max_pool2d_tensor_intarrayref(self.get(), output_size.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor(grad_output.get(), self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref(out.get(), indices.get(), self.get(), output_size.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_adaptive_max_pool3d_tensor_intarrayref(self.get(), output_size.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor(grad_input.get(), grad_output.get(), self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_adaptive_max_pool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor(grad_output.get(), self.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(out.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(out.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_avg_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchbool ceil_mode, XPtrTorchbool count_include_pad, XPtrTorchoptional_int64_t divisor_override) {
  auto r_out = lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), ceil_mode.get(), count_include_pad.get(), divisor_override.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (XPtrTorchTensor output, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchTensor random_samples) {
  auto r_out = lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(output.get(), indices.get(), self.get(), kernel_size.get(), output_size.get(), random_samples.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchTensor random_samples) {
  auto r_out = lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor(self.get(), kernel_size.get(), output_size.get(), random_samples.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), output_size.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor(grad_output.get(), self.get(), kernel_size.get(), output_size.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (XPtrTorchTensor output, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchTensor random_samples) {
  auto r_out = lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(output.get(), indices.get(), self.get(), kernel_size.get(), output_size.get(), random_samples.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchTensor random_samples) {
  auto r_out = lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor(self.get(), kernel_size.get(), output_size.get(), random_samples.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), output_size.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fractional_max_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef output_size, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor(grad_output.get(), self.get(), kernel_size.get(), output_size.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(out.get(), indices.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool2d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchIndexTensor indices, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(out.get(), indices.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_self_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode) {
  auto r_out = lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_input.get(), grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_pool3d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, XPtrTorchbool ceil_mode, XPtrTorchIndexTensor indices) {
  auto r_out = lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(grad_output.get(), self.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), ceil_mode.get(), indices.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref(out.get(), self.get(), indices.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_self_Tensor_indices_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_max_unpool2d_tensor_tensor_intarrayref(self.get(), indices.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), indices.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size) {
  auto r_out = lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref(grad_output.get(), self.get(), indices.get(), output_size.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(out.get(), self.get(), indices.get(), output_size.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref(self.get(), indices.get(), output_size.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(grad_input.get(), grad_output.get(), self.get(), indices.get(), output_size.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_max_unpool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIndexTensor indices, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(grad_output.get(), self.get(), indices.get(), output_size.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad1d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad1d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad1d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad2d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad2d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad2d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad3d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad3d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad3d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad3d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad3d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_reflection_pad3d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_reflection_pad3d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad1d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad1d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad1d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad2d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad2d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad2d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_out_out_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad3d_out_tensor_tensor_intarrayref(out.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_self_Tensor_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad3d_tensor_intarrayref(self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref(grad_input.get(), grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_replication_pad3d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_replication_pad3d_backward_tensor_tensor_intarrayref(grad_output.get(), self.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_linear1d_tensor_intarrayref_bool_arrayrefdouble(input.get(), output_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_bilinear2d_tensor_intarrayref_bool_arrayrefdouble(input.get(), output_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_trilinear3d_tensor_intarrayref_bool_arrayrefdouble(input.get(), output_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_input_Tensor_output_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_bicubic2d_tensor_intarrayref_bool_arrayrefdouble(input.get(), output_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest1d_tensor_intarrayref_arrayrefdouble(input.get(), output_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest2d_tensor_intarrayref_arrayrefdouble(input.get(), output_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_input_Tensor_output_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor input, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest3d_tensor_intarrayref_arrayrefdouble(input.get(), output_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_scale_factors_ArrayRefdouble (XPtrTorchTensor grad_output, XPtrTorchOptionalIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionalDoubleArrayRef scale_factors) {
  auto r_out = lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(grad_output.get(), output_size.get(), input_size.get(), scale_factors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool_double(out.get(), self.get(), output_size.get(), align_corners.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_linear1d_tensor_intarrayref_bool_double(self.get(), output_size.get(), align_corners.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_linear1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_double(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool_double_double(out.get(), self.get(), output_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bilinear2d_tensor_intarrayref_bool_double_double(self.get(), output_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bilinear2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_double_double(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool_double_double(out.get(), self.get(), output_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bicubic2d_tensor_intarrayref_bool_double_double(self.get(), output_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_bicubic2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_double_double(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool_double_double_double(out.get(), self.get(), output_size.get(), align_corners.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_self_Tensor_output_size_IntArrayRef_align_corners_bool (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_trilinear3d_tensor_intarrayref_bool_double_double_double(self.get(), output_size.get(), align_corners.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_trilinear3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchbool align_corners, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_double_double_double(grad_output.get(), output_size.get(), input_size.get(), align_corners.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_nearest1d_out_tensor_tensor_intarrayref_double(out.get(), self.get(), output_size.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_nearest1d_tensor_intarrayref_double(self.get(), output_size.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales) {
  auto r_out = lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_double(grad_output.get(), output_size.get(), input_size.get(), scales.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest2d_out_tensor_tensor_intarrayref_double_double(out.get(), self.get(), output_size.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest2d_tensor_intarrayref_double_double(self.get(), output_size.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_double_double(grad_output.get(), output_size.get(), input_size.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest3d_out_tensor_tensor_intarrayref_double_double_double(out.get(), self.get(), output_size.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_self_Tensor_output_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest3d_tensor_intarrayref_double_double_double(self.get(), output_size.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double_double(grad_input.get(), grad_output.get(), output_size.get(), input_size.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_upsample_nearest3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef input_size, XPtrTorchOptionaldouble scales_d, XPtrTorchOptionaldouble scales_h, XPtrTorchOptionaldouble scales_w) {
  auto r_out = lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_double_double_double(grad_output.get(), output_size.get(), input_size.get(), scales_d.get(), scales_h.get(), scales_w.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor output) {
  auto r_out = lantern_sigmoid_backward_out_tensor_tensor_tensor(grad_input.get(), grad_output.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_sigmoid_backward_grad_output_Tensor_output_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output) {
  auto r_out = lantern_sigmoid_backward_tensor_tensor(grad_output.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_logit_backward_out_tensor_tensor_tensor_double(grad_input.get(), grad_output.get(), self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_logit_backward_grad_output_Tensor_self_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_logit_backward_tensor_tensor_double(grad_output.get(), self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchTensor output) {
  auto r_out = lantern_tanh_backward_out_tensor_tensor_tensor(grad_input.get(), grad_output.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_tanh_backward_grad_output_Tensor_output_Tensor (XPtrTorchTensor grad_output, XPtrTorchTensor output) {
  auto r_out = lantern_tanh_backward_tensor_tensor(grad_output.get(), output.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_bias, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation, XPtrTorchTensor columns, XPtrTorchTensor ones) {
  auto r_out = lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input.get(), grad_weight.get(), grad_bias.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), output_padding.get(), dilation.get(), columns.get(), ones.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation, XPtrTorchTensor columns, XPtrTorchTensor ones, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), output_padding.get(), dilation.get(), columns.get(), ones.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_transpose3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), output_padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_bias, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation, XPtrTorchTensor finput, XPtrTorchTensor fgrad_input) {
  auto r_out = lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input.get(), grad_weight.get(), grad_bias.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), output_padding.get(), dilation.get(), finput.get(), fgrad_input.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef output_padding, XPtrTorchIntArrayRef dilation, XPtrTorchTensor finput, XPtrTorchTensor fgrad_input, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), output_padding.get(), dilation.get(), finput.get(), fgrad_input.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(out.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_thnn_conv2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__slow_conv2d_forward_out_output_Tensor_finput_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor output, XPtrTorchTensor finput, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern__slow_conv2d_forward_out_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(output.get(), finput.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__slow_conv2d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern__slow_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__slow_conv2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_bias, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchTensor finput) {
  auto r_out = lantern__slow_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor(grad_input.get(), grad_weight.get(), grad_bias.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), finput.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__slow_conv2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchTensor finput, std::vector<bool> output_mask) {
  auto r_out = lantern__slow_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), finput.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__conv_depthwise2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern__conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(out.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__conv_depthwise2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern__conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__conv_depthwise2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern__conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_input.get(), grad_weight.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__conv_depthwise2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool2 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, std::vector<bool> output_mask) {
  auto r_out = lantern__conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_conv_depthwise3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_conv_depthwise3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_conv_depthwise3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_bias, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_conv_depthwise3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_input.get(), grad_weight.get(), grad_bias.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_conv_depthwise3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_conv_depthwise3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_slow_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(out.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_slow_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_out_output_Tensor_finput_Tensor_fgrad_input_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor output, XPtrTorchTensor finput, XPtrTorchTensor fgrad_input, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_slow_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(output.get(), finput.get(), fgrad_input.get(), self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding) {
  auto r_out = lantern_slow_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor (XPtrTorchTensor grad_input, XPtrTorchTensor grad_weight, XPtrTorchTensor grad_bias, XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchTensor finput, XPtrTorchTensor fgrad_input) {
  auto r_out = lantern_slow_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(grad_input.get(), grad_weight.get(), grad_bias.get(), grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), finput.get(), fgrad_input.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchTensor finput, XPtrTorchTensor fgrad_input, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), finput.get(), fgrad_input.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_dilated2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_slow_conv_dilated3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchOptionalTensor bias, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation) {
  auto r_out = lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(self.get(), weight.get(), kernel_size.get(), bias.get(), stride.get(), padding.get(), dilation.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (XPtrTorchTensor grad_output, XPtrTorchTensor self, XPtrTorchTensor weight, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef stride, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef dilation, std::vector<bool> output_mask) {
  auto r_out = lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(grad_output.get(), self.get(), weight.get(), kernel_size.get(), stride.get(), padding.get(), dilation.get(), reinterpret_cast<void*>(&output_mask));
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_out_out_Tensor_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(out.get(), self.get(), output_size.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef output_size, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(self.get(), output_size.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_backward_out_grad_input_Tensor_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_input.get(), grad_output.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_col2im_backward_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref(grad_output.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_column_stack_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_column_stack_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_column_stack_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_column_stack_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(out.get(), self.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor self, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref(self.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_backward_out_grad_input_Tensor_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor grad_input, XPtrTorchTensor grad_output, XPtrTorchIntArrayRef input_size, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(grad_input.get(), grad_output.get(), input_size.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_im2col_backward_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (XPtrTorchTensor grad_output, XPtrTorchIntArrayRef input_size, XPtrTorchIntArrayRef kernel_size, XPtrTorchIntArrayRef dilation, XPtrTorchIntArrayRef padding, XPtrTorchIntArrayRef stride) {
  auto r_out = lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(grad_output.get(), input_size.get(), kernel_size.get(), dilation.get(), padding.get(), stride.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isfinite_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isfinite_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isinf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isinf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isposinf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isposinf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isposinf_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_isposinf_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isneginf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_isneginf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_isneginf_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_isneginf_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__add_batch_dim_self_Tensor_batch_dim_int64_t_level_int64_t (XPtrTorchTensor self, XPtrTorchint64_t batch_dim, XPtrTorchint64_t level) {
  auto r_out = lantern__add_batch_dim_tensor_intt_intt(self.get(), batch_dim.get(), level.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__remove_batch_dim_self_Tensor_level_int64_t_batch_size_int64_t_out_dim_int64_t (XPtrTorchTensor self, XPtrTorchint64_t level, XPtrTorchint64_t batch_size, XPtrTorchint64_t out_dim) {
  auto r_out = lantern__remove_batch_dim_tensor_intt_intt_intt(self.get(), level.get(), batch_size.get(), out_dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_entr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_entr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_entr_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_entr_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_ndtri_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_ndtri_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_ndtri_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_ndtri_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_expm1_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_expm1_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_expm1_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_expm1_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_exp2_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_exp2_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_exp2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_exp2_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_psi_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_psi_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_psi_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_psi_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_digamma_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_digamma_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_digamma_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_digamma_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammaln_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_gammaln_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammaln_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_gammaln_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erf_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_erf_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erf_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_erf_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_erfc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_erfc_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfcx_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_erfcx_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfcx_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_erfcx_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfinv_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_erfinv_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_erfinv_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_erfinv_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_ndtr_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_ndtr_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_ndtr_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_ndtr_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlog1py_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlog1py_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_xlog1py_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlog1py_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_out_out_Tensor_self_Scalar_other_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlog1py_out_tensor_scalar_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlog1py_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_xlog1py_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlogy_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlogy_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_xlogy_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlogy_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_out_out_Tensor_self_Scalar_other_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_xlogy_out_tensor_scalar_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_xlogy_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_xlogy_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_zeta_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_self_Scalar_other_Tensor (XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_zeta_scalar_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_self_Tensor_other_Scalar (XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_zeta_tensor_scalar(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_zeta_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_out_out_Tensor_self_Scalar_other_Tensor (XPtrTorchTensor out, XPtrTorchScalar self, XPtrTorchTensor other) {
  auto r_out = lantern_special_zeta_out_tensor_scalar_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_zeta_out_out_Tensor_self_Tensor_other_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar other) {
  auto r_out = lantern_special_zeta_out_tensor_tensor_scalar(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i0_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_i0_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i0_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_i0_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i0e_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_i0e_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i0e_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_i0e_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i1_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_i1_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i1_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_i1_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i1e_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_i1e_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_i1e_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_i1e_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_logit_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_special_logit_tensor_double(self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_logit_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionaldouble eps) {
  auto r_out = lantern_special_logit_out_tensor_tensor_double(out.get(), self.get(), eps.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_polygamma_out_out_Tensor_n_int64_t_self_Tensor (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchTensor self) {
  auto r_out = lantern_special_polygamma_out_tensor_intt_tensor(out.get(), n.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_logsumexp_self_Tensor_dim_IntArrayRef (XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_special_logsumexp_tensor_intarrayref_bool(self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_logsumexp_out_out_Tensor_self_Tensor_dim_IntArrayRef (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim) {
  auto r_out = lantern_special_logsumexp_out_tensor_tensor_intarrayref_bool(out.get(), self.get(), dim.get(), keepdim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_expit_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_expit_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_expit_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_expit_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_sinc_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_sinc_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_sinc_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_sinc_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_round_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_round_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_round_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_round_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_log1p_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_special_log1p_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_log1p_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_special_log1p_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_log_softmax_self_Tensor_dim_int64_t (XPtrTorchTensor self, XPtrTorchindex_int64_t dim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_special_log_softmax_tensor_intt_scalartype(self.get(), dim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammainc_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_gammainc_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammainc_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_gammainc_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammaincc_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_gammaincc_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_gammaincc_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_special_gammaincc_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_multigammaln_self_Tensor_p_int64_t (XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_special_multigammaln_tensor_intt(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_special_multigammaln_out_out_Tensor_self_Tensor_p_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t p) {
  auto r_out = lantern_special_multigammaln_out_tensor_tensor_intt(out.get(), self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_hfft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_hfft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_hfft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_hfft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ihfft_self_Tensor (XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ihfft_tensor_intt_intt_cstringview(self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ihfft_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_int64_t n, XPtrTorchindex_int64_t dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ihfft_out_tensor_tensor_intt_intt_cstringview(out.get(), self.get(), n.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fft2_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fft2_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fft2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fft2_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifft2_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifft2_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifft2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifft2_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfft2_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfft2_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfft2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfft2_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfft2_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfft2_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfft2_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfft2_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftn_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fftn_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftn_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_fftn_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifftn_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifftn_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifftn_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_ifftn_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfftn_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfftn_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfftn_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_rfftn_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfftn_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfftn_tensor_intarrayref_intarrayref_cstringview(self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_irfftn_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionalIntArrayRef s, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchoptional_string_view norm) {
  auto r_out = lantern_fft_irfftn_out_tensor_tensor_intarrayref_intarrayref_cstringview(out.get(), self.get(), s.get(), dim.get(), norm.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftfreq_n_int64_t (XPtrTorchint64_t n, XPtrTorchdouble d, XPtrTorchTensorOptions options) {
  auto r_out = lantern_fft_fftfreq_intt_double_tensoroptions(n.get(), d.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftfreq_out_out_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchdouble d) {
  auto r_out = lantern_fft_fftfreq_out_tensor_intt_double(out.get(), n.get(), d.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfftfreq_n_int64_t (XPtrTorchint64_t n, XPtrTorchdouble d, XPtrTorchTensorOptions options) {
  auto r_out = lantern_fft_rfftfreq_intt_double_tensoroptions(n.get(), d.get(), options.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_rfftfreq_out_out_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchint64_t n, XPtrTorchdouble d) {
  auto r_out = lantern_fft_rfftfreq_out_tensor_intt_double(out.get(), n.get(), d.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_fftshift_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim) {
  auto r_out = lantern_fft_fftshift_tensor_intarrayref(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_fft_ifftshift_self_Tensor (XPtrTorchTensor self, XPtrTorchOptionalIndexIntArrayRef dim) {
  auto r_out = lantern_fft_ifftshift_tensor_intarrayref(self.get(), dim.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_cholesky_ex_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper, XPtrTorchbool check_errors) {
  auto r_out = lantern_linalg_cholesky_ex_tensor_bool_bool(self.get(), upper.get(), check_errors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_cholesky_ex_out_L_Tensor_info_Tensor_self_Tensor (XPtrTorchTensor L, XPtrTorchTensor info, XPtrTorchTensor self, XPtrTorchbool upper, XPtrTorchbool check_errors) {
  auto r_out = lantern_linalg_cholesky_ex_out_tensor_tensor_tensor_bool_bool(L.get(), info.get(), self.get(), upper.get(), check_errors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cholesky_self_Tensor (XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_linalg_cholesky_tensor_bool(self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cholesky_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchbool upper) {
  auto r_out = lantern_linalg_cholesky_out_tensor_tensor_bool(out.get(), self.get(), upper.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_det_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_linalg_det_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_det_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_linalg_det_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_det_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_det_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__det_lu_based_helper_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern__det_lu_based_helper_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__det_lu_based_helper_backward_helper_det_grad_Tensor_det_Tensor_self_Tensor_lu_Tensor_pivs_Tensor (XPtrTorchTensor det_grad, XPtrTorchTensor det, XPtrTorchTensor self, XPtrTorchTensor lu, XPtrTorchTensor pivs) {
  auto r_out = lantern__det_lu_based_helper_backward_helper_tensor_tensor_tensor_tensor_tensor(det_grad.get(), det.get(), self.get(), lu.get(), pivs.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_lstsq_self_Tensor_b_Tensor (XPtrTorchTensor self, XPtrTorchTensor b, XPtrTorchOptionaldouble rcond, XPtrTorchoptional_string_view driver) {
  auto r_out = lantern_linalg_lstsq_tensor_tensor_double_cstringview(self.get(), b.get(), rcond.get(), driver.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_lstsq_out_solution_Tensor_residuals_Tensor_rank_Tensor_singular_values_Tensor_self_Tensor_b_Tensor (XPtrTorchTensor solution, XPtrTorchTensor residuals, XPtrTorchTensor rank, XPtrTorchTensor singular_values, XPtrTorchTensor self, XPtrTorchTensor b, XPtrTorchOptionaldouble rcond, XPtrTorchoptional_string_view driver) {
  auto r_out = lantern_linalg_lstsq_out_tensor_tensor_tensor_tensor_tensor_tensor_double_cstringview(solution.get(), residuals.get(), rank.get(), singular_values.get(), self.get(), b.get(), rcond.get(), driver.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 3)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matmul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_linalg_matmul_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matmul_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_linalg_matmul_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_slogdet_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_linalg_slogdet_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_slogdet_out_sign_Tensor_logabsdet_Tensor_self_Tensor (XPtrTorchTensor sign, XPtrTorchTensor logabsdet, XPtrTorchTensor self) {
  auto r_out = lantern_linalg_slogdet_out_tensor_tensor_tensor(sign.get(), logabsdet.get(), self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_eig_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_linalg_eig_tensor(self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_eig_out_eigenvalues_Tensor_eigenvectors_Tensor_self_Tensor (XPtrTorchTensor eigenvalues, XPtrTorchTensor eigenvectors, XPtrTorchTensor self) {
  auto r_out = lantern_linalg_eig_out_tensor_tensor_tensor(eigenvalues.get(), eigenvectors.get(), self.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_eigvals_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_linalg_eigvals_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_eigvals_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_linalg_eigvals_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_eigh_self_Tensor (XPtrTorchTensor self, XPtrTorchstring_view UPLO) {
  auto r_out = lantern_linalg_eigh_tensor_cstringview(self.get(), UPLO.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_eigh_out_eigvals_Tensor_eigvecs_Tensor_self_Tensor (XPtrTorchTensor eigvals, XPtrTorchTensor eigvecs, XPtrTorchTensor self, XPtrTorchstring_view UPLO) {
  auto r_out = lantern_linalg_eigh_out_tensor_tensor_tensor_cstringview(eigvals.get(), eigvecs.get(), self.get(), UPLO.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_eigvalsh_self_Tensor (XPtrTorchTensor self, XPtrTorchstring_view UPLO) {
  auto r_out = lantern_linalg_eigvalsh_tensor_cstringview(self.get(), UPLO.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_eigvalsh_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchstring_view UPLO) {
  auto r_out = lantern_linalg_eigvalsh_out_tensor_tensor_cstringview(out.get(), self.get(), UPLO.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_householder_product_input_Tensor_tau_Tensor (XPtrTorchTensor input, XPtrTorchTensor tau) {
  auto r_out = lantern_linalg_householder_product_tensor_tensor(input.get(), tau.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_householder_product_out_out_Tensor_input_Tensor_tau_Tensor (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchTensor tau) {
  auto r_out = lantern_linalg_householder_product_out_tensor_tensor_tensor(out.get(), input.get(), tau.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__linalg_inv_out_helper__self_Tensor_infos_lu_Tensor_infos_getri_Tensor (XPtrTorchTensor self, XPtrTorchTensor infos_lu, XPtrTorchTensor infos_getri) {
  auto r_out = lantern__linalg_inv_out_helper__tensor_tensor_tensor(self.get(), infos_lu.get(), infos_getri.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_inv_ex_self_Tensor (XPtrTorchTensor self, XPtrTorchbool check_errors) {
  auto r_out = lantern_linalg_inv_ex_tensor_bool(self.get(), check_errors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_inv_ex_out_inverse_Tensor_info_Tensor_self_Tensor (XPtrTorchTensor inverse, XPtrTorchTensor info, XPtrTorchTensor self, XPtrTorchbool check_errors) {
  auto r_out = lantern_linalg_inv_ex_out_tensor_tensor_tensor_bool(inverse.get(), info.get(), self.get(), check_errors.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_inv_self_Tensor (XPtrTorchTensor self) {
  auto r_out = lantern_linalg_inv_tensor(self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_inv_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self) {
  auto r_out = lantern_linalg_inv_out_tensor_tensor(out.get(), self.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inner_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_inner_tensor_tensor(self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_inner_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other) {
  auto r_out = lantern_inner_out_tensor_tensor_tensor(out.get(), self.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_outer_self_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_outer_tensor_tensor(self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_outer_out_out_Tensor_self_Tensor_vec2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_outer_out_tensor_tensor_tensor(out.get(), self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ger_self_Tensor_vec2_Tensor (XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_ger_tensor_tensor(self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_ger_out_out_Tensor_self_Tensor_vec2_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor vec2) {
  auto r_out = lantern_ger_out_tensor_tensor_tensor(out.get(), self.get(), vec2.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_self_Tensor_ord_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_self_Tensor_ord_c10string_view (XPtrTorchTensor self, XPtrTorchstring_view ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_norm_tensor_cstringview_intarrayref_bool_scalartype(self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_out_out_Tensor_self_Tensor_ord_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out.get(), self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_norm_out_out_Tensor_self_Tensor_ord_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchstring_view ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_norm_out_tensor_tensor_cstringview_intarrayref_bool_scalartype(out.get(), self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_vector_norm_self_Tensor (XPtrTorchTensor self, XPtrTorchScalar ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_vector_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_vector_norm_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar ord, XPtrTorchOptionalIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_vector_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out.get(), self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_norm_self_Tensor_ord_Scalar (XPtrTorchTensor self, XPtrTorchScalar ord, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_matrix_norm_tensor_scalar_intarrayref_bool_scalartype(self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_norm_out_out_Tensor_self_Tensor_ord_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchScalar ord, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_matrix_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(out.get(), self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_norm_self_Tensor_ord_c10string_view (XPtrTorchTensor self, XPtrTorchstring_view ord, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_matrix_norm_tensor_cstringview_intarrayref_bool_scalartype(self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_norm_out_out_Tensor_self_Tensor_ord_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchstring_view ord, XPtrTorchIndexIntArrayRef dim, XPtrTorchbool keepdim, XPtrTorchoptional_scalar_type dtype) {
  auto r_out = lantern_linalg_matrix_norm_out_tensor_tensor_cstringview_intarrayref_bool_scalartype(out.get(), self.get(), ord.get(), dim.get(), keepdim.get(), dtype.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_svd_out_U_Tensor_S_Tensor_Vh_Tensor_self_Tensor (XPtrTorchTensor U, XPtrTorchTensor S, XPtrTorchTensor Vh, XPtrTorchTensor self, XPtrTorchbool full_matrices) {
  auto r_out = lantern_linalg_svd_out_tensor_tensor_tensor_tensor_bool(U.get(), S.get(), Vh.get(), self.get(), full_matrices.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_svd_self_Tensor (XPtrTorchTensor self, XPtrTorchbool full_matrices) {
  auto r_out = lantern_linalg_svd_tensor_bool(self.get(), full_matrices.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 2)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_svdvals_input_Tensor (XPtrTorchTensor input) {
  auto r_out = lantern_linalg_svdvals_tensor(input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_svdvals_out_out_Tensor_input_Tensor (XPtrTorchTensor out, XPtrTorchTensor input) {
  auto r_out = lantern_linalg_svdvals_out_tensor_tensor(out.get(), input.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cond_self_Tensor_p_Scalar (XPtrTorchTensor self, XPtrTorchoptional_scalar p) {
  auto r_out = lantern_linalg_cond_tensor_scalar(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cond_out_out_Tensor_self_Tensor_p_Scalar (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchoptional_scalar p) {
  auto r_out = lantern_linalg_cond_out_tensor_tensor_scalar(out.get(), self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cond_self_Tensor_p_c10string_view (XPtrTorchTensor self, XPtrTorchstring_view p) {
  auto r_out = lantern_linalg_cond_tensor_cstringview(self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_cond_out_out_Tensor_self_Tensor_p_c10string_view (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchstring_view p) {
  auto r_out = lantern_linalg_cond_out_tensor_tensor_cstringview(out.get(), self.get(), p.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_pinv_self_Tensor_rcond_double (XPtrTorchTensor self, XPtrTorchdouble rcond, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_pinv_tensor_double_bool(self.get(), rcond.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_pinv_self_Tensor_rcond_Tensor (XPtrTorchTensor self, XPtrTorchTensor rcond, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_pinv_tensor_tensor_bool(self.get(), rcond.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_pinv_out_out_Tensor_self_Tensor_rcond_double (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchdouble rcond, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_pinv_out_tensor_tensor_double_bool(out.get(), self.get(), rcond.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_pinv_out_out_Tensor_self_Tensor_rcond_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor rcond, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_pinv_out_tensor_tensor_tensor_bool(out.get(), self.get(), rcond.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_solve_input_Tensor_other_Tensor (XPtrTorchTensor input, XPtrTorchTensor other) {
  auto r_out = lantern_linalg_solve_tensor_tensor(input.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_solve_out_out_Tensor_input_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchTensor other) {
  auto r_out = lantern_linalg_solve_out_tensor_tensor_tensor(out.get(), input.get(), other.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_tensorinv_self_Tensor (XPtrTorchTensor self, XPtrTorchint64_t ind) {
  auto r_out = lantern_linalg_tensorinv_tensor_intt(self.get(), ind.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_tensorinv_out_out_Tensor_self_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t ind) {
  auto r_out = lantern_linalg_tensorinv_out_tensor_tensor_intt(out.get(), self.get(), ind.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_tensorsolve_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchOptionalIndexIntArrayRef dims) {
  auto r_out = lantern_linalg_tensorsolve_tensor_tensor_intarrayref(self.get(), other.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_tensorsolve_out_out_Tensor_self_Tensor_other_Tensor (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchOptionalIndexIntArrayRef dims) {
  auto r_out = lantern_linalg_tensorsolve_out_tensor_tensor_tensor_intarrayref(out.get(), self.get(), other.get(), dims.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_qr_self_Tensor (XPtrTorchTensor self, XPtrTorchstring_view mode) {
  auto r_out = lantern_linalg_qr_tensor_cstringview(self.get(), mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_linalg_qr_out_Q_Tensor_R_Tensor_self_Tensor (XPtrTorchTensor Q, XPtrTorchTensor R, XPtrTorchTensor self, XPtrTorchstring_view mode) {
  auto r_out = lantern_linalg_qr_out_tensor_tensor_tensor_cstringview(Q.get(), R.get(), self.get(), mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__linalg_qr_helper_self_Tensor_mode_c10string_view (XPtrTorchTensor self, XPtrTorchstring_view mode) {
  auto r_out = lantern__linalg_qr_helper_tensor_cstringview(self.get(), mode.get());
auto wrap = XPtrTorchvector_void(r_out);
return Rcpp::List::create(XPtrTorchTensor(lantern_vector_get(wrap.get(), 0)),XPtrTorchTensor(lantern_vector_get(wrap.get(), 1)));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_power_self_Tensor_n_int64_t (XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_linalg_matrix_power_tensor_intt(self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_power_out_out_Tensor_self_Tensor_n_int64_t (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchint64_t n) {
  auto r_out = lantern_linalg_matrix_power_out_tensor_tensor_intt(out.get(), self.get(), n.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_rank_self_Tensor_tol_double (XPtrTorchTensor self, XPtrTorchOptionaldouble tol, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_matrix_rank_tensor_double_bool(self.get(), tol.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_rank_out_out_Tensor_self_Tensor_tol_double (XPtrTorchTensor out, XPtrTorchTensor self, XPtrTorchOptionaldouble tol, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_matrix_rank_out_tensor_tensor_double_bool(out.get(), self.get(), tol.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_rank_input_Tensor_tol_Tensor (XPtrTorchTensor input, XPtrTorchTensor tol, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_matrix_rank_tensor_tensor_bool(input.get(), tol.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_matrix_rank_out_out_Tensor_input_Tensor_tol_Tensor (XPtrTorchTensor out, XPtrTorchTensor input, XPtrTorchTensor tol, XPtrTorchbool hermitian) {
  auto r_out = lantern_linalg_matrix_rank_out_tensor_tensor_tensor_bool(out.get(), input.get(), tol.get(), hermitian.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_multi_dot_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_linalg_multi_dot_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_linalg_multi_dot_out_out_Tensor_tensors_TensorList (XPtrTorchTensor out, XPtrTorchTensorList tensors) {
  auto r_out = lantern_linalg_multi_dot_out_tensor_tensorlist(out.get(), tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_serialization_subcmul_self_Tensor_other_Tensor (XPtrTorchTensor self, XPtrTorchTensor other, XPtrTorchScalar alpha) {
  auto r_out = lantern__test_serialization_subcmul_tensor_tensor_scalar(self.get(), other.get(), alpha.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_intlist_values_Tensor_addends_IntArrayRef (XPtrTorchTensor values, XPtrTorchOptionalIntArrayRef addends) {
  auto r_out = lantern__test_optional_intlist_tensor_intarrayref(values.get(), addends.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_filled_intlist_values_Tensor_addends_IntArrayRef (XPtrTorchTensor values, XPtrTorchOptionalIntArrayRef addends) {
  auto r_out = lantern__test_optional_filled_intlist_tensor_intarrayref(values.get(), addends.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__test_optional_floatlist_values_Tensor_addends_ArrayRefdouble (XPtrTorchTensor values, XPtrTorchOptionalDoubleArrayRef addends) {
  auto r_out = lantern__test_optional_floatlist_tensor_arrayrefdouble(values.get(), addends.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_segment_reduce_data_Tensor_reduce_c10string_view (XPtrTorchTensor data, XPtrTorchstring_view reduce, XPtrTorchOptionalTensor lengths, XPtrTorchIndexTensor indices, XPtrTorchint64_t axis, XPtrTorchbool unsafe, XPtrTorchoptional_scalar initial) {
  auto r_out = lantern_segment_reduce_tensor_cstringview_tensor_tensor_intt_bool_scalar(data.get(), reduce.get(), lengths.get(), indices.get(), axis.get(), unsafe.get(), initial.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace__segment_reduce_backward_grad_Tensor_output_Tensor_data_Tensor_reduce_c10string_view (XPtrTorchTensor grad, XPtrTorchTensor output, XPtrTorchTensor data, XPtrTorchstring_view reduce, XPtrTorchOptionalTensor lengths, XPtrTorchint64_t axis) {
  auto r_out = lantern__segment_reduce_backward_tensor_tensor_tensor_cstringview_tensor_intt(grad.get(), output.get(), data.get(), reduce.get(), lengths.get(), axis.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_pad_sequence_sequences_TensorList (XPtrTorchTensorList sequences, XPtrTorchbool batch_first, XPtrTorchdouble padding_value) {
  auto r_out = lantern_pad_sequence_tensorlist_bool_double(sequences.get(), batch_first.get(), padding_value.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_namespace_flatten_dense_tensors_tensors_TensorList (XPtrTorchTensorList tensors) {
  auto r_out = lantern_flatten_dense_tensors_tensorlist(tensors.get());
return XPtrTorchTensor(r_out);
}

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_namespace_unflatten_dense_tensors_flat_Tensor_tensors_TensorList (XPtrTorchTensor flat, XPtrTorchTensorList tensors) {
  auto r_out = lantern_unflatten_dense_tensors_tensor_tensorlist(flat.get(), tensors.get());
return XPtrTorchTensorList(r_out);
}

