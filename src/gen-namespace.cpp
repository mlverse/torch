#include "torch_types.h"
#include "utils.hpp"

      // Workaround for https://discuss.pytorch.org/t/using-torch-normal-with-the-c-frontend/68971?u=dfalbel
      namespace torch {

      torch::Tensor normal (const torch::Tensor &mean, double std = 1, torch::Generator *generator = nullptr) {
        return at::normal(mean, std, generator);
      }

      torch::Tensor normal (double mean, const torch::Tensor &std, torch::Generator *generator = nullptr) {
        return at::normal(mean, std, generator);
      }

      torch::Tensor normal (const torch::Tensor &mean, const torch::Tensor &std, torch::Generator *generator = nullptr) {
        return at::normal(mean, std, generator);
      }

      } // namespace torch
      
// [[Rcpp::export]]
void cpp_torch_method_backward_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> gradient, bool keep_graph, bool create_graph) {
  self->backward(* gradient, keep_graph, create_graph);
}

// [[Rcpp::export]]
void cpp_torch_method_set_data_self_Tensor_new_data_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> new_data) {
  self->set_data(* new_data);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_data_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->data();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_leaf_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_leaf();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method_output_nr_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->output_nr();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method__version_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_version();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_requires_grad__self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool requires_grad) {
  auto r_out = self->requires_grad_(requires_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rename__self_Tensor_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->rename_(* names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rename_self_Tensor_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->rename(* names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_align_to_self_Tensor_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->align_to(* names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_align_to_self_Tensor_order_DimnameList_ellipsis_idx_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> order, int64_t ellipsis_idx) {
  auto r_out = self->align_to(* order, ellipsis_idx);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_align_as_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->align_as(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_refine_names_self_Tensor_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->refine_names(* names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_unflatten_self_Tensor_dim_Dimname_sizes_IntArrayRef_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, std::vector<int64_t> sizes, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->unflatten(* dim, sizes, * names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_unflatten_self_Tensor_dim_int64_t_sizes_IntArrayRef_names_DimnameList (Rcpp::XPtr<torch::Tensor> self, int64_t dim, std::vector<int64_t> sizes, Rcpp::XPtr<std::vector<torch::Dimname>> names) {
  auto r_out = self->unflatten(dim, sizes, * names);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_abs_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->abs();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_abs__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->abs_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_angle_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->angle();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_real_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->real();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_imag_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->imag();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_conj_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->conj();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_acos_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->acos();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_acos__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->acos_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_add_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->add(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_add__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->add_(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_add_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->add(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_add__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->add_(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addmv_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat, Rcpp::XPtr<torch::Tensor> vec, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addmv(* mat, * vec, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addmv__self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat, Rcpp::XPtr<torch::Tensor> vec, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addmv_(* mat, * vec, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addr_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addr(* vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addr__self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addr_(* vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_all_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->all(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_all_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->all(* dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_allclose_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = self->allclose(* other, rtol, atol, equal_nan);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_any_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->any(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_any_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->any(* dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_argmax_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->argmax(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_argmin_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->argmin(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, int64_t storage_offset) {
  auto r_out = self->as_strided(size, stride, storage_offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, int64_t storage_offset) {
  auto r_out = self->as_strided_(size, stride, storage_offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_asin_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->asin();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_asin__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->asin_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_atan_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->atan();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_atan__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->atan_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->baddbmm(* batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_baddbmm__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->baddbmm_(* batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bernoulli_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->bernoulli(* generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bernoulli__self_Tensor_p_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->bernoulli_(* p, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bernoulli__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->bernoulli_(p, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bernoulli_self_Tensor_p_double (Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->bernoulli(p, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bincount_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weights, int64_t minlength) {
  auto r_out = self->bincount(* weights, minlength);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_not_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->bitwise_not();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_not__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->bitwise_not_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logical_not_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->logical_not();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logical_not__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->logical_not_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logical_xor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->logical_xor(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logical_xor__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->logical_xor_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bmm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = self->bmm(* mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ceil_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->ceil();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ceil__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->ceil_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t chunks, int64_t dim) {
  auto r_out = self->chunk(chunks, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = self->clamp(* min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = self->clamp_(* min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp_max_self_Tensor_max_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = self->clamp_max(* max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp_max__self_Tensor_max_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = self->clamp_max_(* max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp_min_self_Tensor_min_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min) {
  auto r_out = self->clamp_min(* min);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clamp_min__self_Tensor_min_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min) {
  auto r_out = self->clamp_min_(* min);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_contiguous_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->contiguous(* memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_copy__self_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> src, bool non_blocking) {
  auto r_out = self->copy_(* src, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cos_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->cos();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cos__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->cos_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cosh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->cosh();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cosh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->cosh_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->cumsum(dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cumsum_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->cumsum(* dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->cumprod(dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cumprod_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->cumprod(* dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_det_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->det();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_diag_embed_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto r_out = self->diag_embed(offset, dim1, dim2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_diagflat_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset) {
  auto r_out = self->diagflat(offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_diagonal_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto r_out = self->diagonal(offset, dim1, dim2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fill_diagonal__self_Tensor_fill_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> fill_value, bool wrap) {
  auto r_out = self->fill_diagonal_(* fill_value, wrap);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_div_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->div(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_div__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->div_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_div_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->div(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_div__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->div_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_dot_self_Tensor_tensor_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor) {
  auto r_out = self->dot(* tensor);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_new_empty_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = self->new_empty(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_new_full_self_Tensor_size_IntArrayRef_fill_value_Scalar (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, Rcpp::XPtr<torch::Scalar> fill_value, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = self->new_full(size, * fill_value, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_new_zeros_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = self->new_zeros(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_resize__self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->resize_(size, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erf_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erf();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erf__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erf_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erfc_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erfc();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erfc__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erfc_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_exp_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->exp();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_exp__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->exp_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_expm1_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->expm1();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_expm1__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->expm1_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_expand_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, bool implicit) {
  auto r_out = self->expand(size, implicit);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_expand_as_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->expand_as(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_flatten_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t start_dim, int64_t end_dim) {
  auto r_out = self->flatten(start_dim, end_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, int64_t start_dim, int64_t end_dim, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = self->flatten(start_dim, end_dim, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> start_dim, Rcpp::XPtr<torch::Dimname> end_dim, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = self->flatten(* start_dim, * end_dim, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dims, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = self->flatten(* dims, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fill__self_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->fill_(* value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fill__self_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->fill_(* value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_floor_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->floor();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_floor__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->floor_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_frac_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->frac();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_frac__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->frac_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ger_self_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec2) {
  auto r_out = self->ger(* vec2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized) {
  auto r_out = self->fft(signal_ndim, normalized);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ifft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized) {
  auto r_out = self->ifft(signal_ndim, normalized);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized, bool onesided) {
  auto r_out = self->rfft(signal_ndim, normalized, onesided);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_irfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized, bool onesided, std::vector<int64_t> signal_sizes) {
  auto r_out = self->irfft(signal_ndim, normalized, onesided, signal_sizes);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_self_Tensor_indices_TensorList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices) {
  auto r_out = self->index(* indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_copy_(dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_copy(dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_copy__self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_copy_(* dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_copy(* dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_put__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices, Rcpp::XPtr<torch::Tensor> values, bool accumulate) {
  auto r_out = self->index_put_(* indices, * values, accumulate);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_put_self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices, Rcpp::XPtr<torch::Tensor> values, bool accumulate) {
  auto r_out = self->index_put(* indices, * values, accumulate);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_inverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->inverse();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_isclose_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = self->isclose(* other, rtol, atol, equal_nan);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_distributed_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_distributed();
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_method_is_floating_point_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_floating_point();
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_method_is_complex_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_complex();
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_method_is_nonzero_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_nonzero();
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_method_is_same_size_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->is_same_size(* other);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_method_is_signed_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_signed();
return r_out;
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool keepdim) {
  auto r_out = self->kthvalue(k, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_kthvalue_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, int64_t k, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->kthvalue(k, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log10_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log10();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log10__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log10_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log1p_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log1p();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log1p__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log1p_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log2_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log2();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log2__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->log2_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logdet_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->logdet();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->log_softmax(dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->log_softmax(* dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logsumexp_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = self->logsumexp(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_logsumexp_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = self->logsumexp(* dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_matmul_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->matmul(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_matrix_power_self_Tensor_n_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n) {
  auto r_out = self->matrix_power(n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->max(dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_max_values_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = self->max_values(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_max_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->max(* dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_max_values_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = self->max_values(* dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mean_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->mean(* dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->mean(dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->mean(* dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->median(dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_median_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->median(* dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->min(dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_min_values_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = self->min_values(dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_min_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->min(* dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_min_values_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = self->min_values(* dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = self->mm(* mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = self->mode(dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_mode_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = self->mode(* dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mul_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->mul(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mul__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->mul_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mul_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->mul(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mul__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->mul_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mv_self_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec) {
  auto r_out = self->mv(* vec);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mvlgamma_self_Tensor_p_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t p) {
  auto r_out = self->mvlgamma(p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_mvlgamma__self_Tensor_p_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t p) {
  auto r_out = self->mvlgamma_(p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_narrow_copy_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t start, int64_t length) {
  auto r_out = self->narrow_copy(dim, start, length);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t start, int64_t length) {
  auto r_out = self->narrow(dim, start, length);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_permute_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dims) {
  auto r_out = self->permute(dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_numpy_T_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->numpy_T();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_pinned_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_pinned();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pin_memory_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->pin_memory();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pinverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self, double rcond) {
  auto r_out = self->pinverse(rcond);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_reciprocal_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->reciprocal();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_reciprocal__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->reciprocal_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_neg_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->neg();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_neg__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->neg_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_repeat_self_Tensor_repeats_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> repeats) {
  auto r_out = self->repeat(repeats);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_repeat_interleave_self_Tensor_repeats_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> repeats, int64_t dim) {
  auto r_out = self->repeat_interleave(* repeats, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_repeat_interleave_self_Tensor_repeats_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t repeats, int64_t dim) {
  auto r_out = self->repeat_interleave(repeats, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> shape) {
  auto r_out = self->reshape(shape);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_reshape_as_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->reshape_as(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_round_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->round();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_round__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->round_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_relu_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->relu();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_relu__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->relu_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_prelu_self_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = self->prelu(* weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = self->prelu_backward(* grad_output, * weight);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_hardshrink_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = self->hardshrink(* lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = self->hardshrink_backward(* grad_out, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rsqrt_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->rsqrt();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rsqrt__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->rsqrt_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_select_self_Tensor_dim_Dimname_index_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, int64_t index) {
  auto r_out = self->select(* dim, index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_select_self_Tensor_dim_int64_t_index_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t index) {
  auto r_out = self->select(dim, index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sigmoid_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sigmoid();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sigmoid__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sigmoid_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sin_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sin();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sin__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sin_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sinh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sinh();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sinh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sinh_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_detach_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->detach();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_detach__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->detach_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_size_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->size(dim);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method_size_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = self->size(* dim);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_slice_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto r_out = self->slice(dim, start, end, step);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_slogdet_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->slogdet();
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_smm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = self->smm(* mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->softmax(dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->softmax(* dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t split_size, int64_t dim) {
  auto r_out = self->split(split_size, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> split_sizes, int64_t dim) {
  auto r_out = self->split_with_sizes(split_sizes, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->squeeze();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->squeeze(dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = self->squeeze(* dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->squeeze_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze__self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->squeeze_(dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_squeeze__self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = self->squeeze_(* dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->sspaddmm(* mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_stft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n_fft, int64_t hop_length, int64_t win_length, Rcpp::XPtr<torch::Tensor> window, bool normalized, bool onesided) {
  auto r_out = self->stft(n_fft, hop_length, win_length, * window, normalized, onesided);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_stride_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->stride(dim);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method_stride_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = self->stride(* dim);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sum_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->sum(* dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->sum(dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sum_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->sum(* dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sum_to_size_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size) {
  auto r_out = self->sum_to_size(size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sqrt_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sqrt();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sqrt__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sqrt_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_std_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = self->std(unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_std_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = self->std(dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_std_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = self->std(* dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_prod_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->prod(* dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_prod_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->prod(dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_prod_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->prod(* dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_t_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->t();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_t__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->t_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tan_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->tan();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tan__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->tan_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tanh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->tanh();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tanh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->tanh_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim0, int64_t dim1) {
  auto r_out = self->transpose(dim0, dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim0, Rcpp::XPtr<torch::Dimname> dim1) {
  auto r_out = self->transpose(* dim0, * dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim0, int64_t dim1) {
  auto r_out = self->transpose_(dim0, dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_flip_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dims) {
  auto r_out = self->flip(dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_roll_self_Tensor_shifts_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
  auto r_out = self->roll(shifts, dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_rot90_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t k, std::vector<int64_t> dims) {
  auto r_out = self->rot90(k, dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_trunc_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->trunc();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_trunc__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->trunc_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_type_as_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->type_as(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_unsqueeze_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->unsqueeze(dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_unsqueeze__self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->unsqueeze_(dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_var_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = self->var(unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_var_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = self->var(dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_var_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = self->var(* dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_view_as_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->view_as(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> condition, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->where(* condition, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor_p_Scalar_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->norm(* p, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p) {
  auto r_out = self->norm(* p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->norm(* p, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = self->norm(* p, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = self->norm(* p, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_norm_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = self->norm(* p, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_clone_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->clone(* memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_resize_as__self_Tensor_the_template_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> the_template, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->resize_as_(* the_template, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pow_self_Tensor_exponent_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> exponent) {
  auto r_out = self->pow(* exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_zero__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->zero_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sub_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->sub(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sub__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->sub_(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sub_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->sub(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sub__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->sub_(* other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addmm(* mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addmm__self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addmm_(* mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sparse_resize__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  auto r_out = self->sparse_resize_(size, sparse_dim, dense_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sparse_resize_and_clear__self_Tensor_size_IntArrayRef_sparse_dim_int64_t_dense_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  auto r_out = self->sparse_resize_and_clear_(size, sparse_dim, dense_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sparse_mask_self_Tensor_mask_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask) {
  auto r_out = self->sparse_mask(* mask);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_dense_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->to_dense();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_sparse_dim_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sparse_dim();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method__dimI_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_dimI();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method_dense_dim_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->dense_dim();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method__dimV_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_dimV();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method__nnz_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_nnz();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_coalesce_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->coalesce();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_coalesced_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->is_coalesced();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method__indices_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_indices();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method__values_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->_values();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method__coalesced__self_Tensor_coalesced_bool (Rcpp::XPtr<torch::Tensor> self, bool coalesced) {
  auto r_out = self->_coalesced_(coalesced);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_indices_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->indices();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_values_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->values();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_unbind_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = self->unbind(dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_unbind_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = self->unbind(* dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_sparse_self_Tensor_sparse_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t sparse_dim) {
  auto r_out = self->to_sparse(sparse_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_sparse_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->to_sparse();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_mkldnn_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->to_mkldnn();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_dequantize_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->dequantize();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
double cpp_torch_method_q_scale_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->q_scale();
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_method_q_zero_point_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->q_zero_point();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_q_per_channel_scales_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->q_per_channel_scales();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_q_per_channel_zero_points_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->q_per_channel_zero_points();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_method_q_per_channel_axis_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->q_per_channel_axis();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_int_repr_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->int_repr();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::QScheme> cpp_torch_method_qscheme_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->qscheme();
return make_xptr<torch::QScheme>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, bool non_blocking, bool copy, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->to(* options, non_blocking, copy, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_self_Tensor_device_Device_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Device> device, Rcpp::XPtr<torch::Dtype> dtype, bool non_blocking, bool copy, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->to(* device, * dtype, non_blocking, copy, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_self_Tensor_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype, bool non_blocking, bool copy, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->to(* dtype, non_blocking, copy, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_to_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, bool non_blocking, bool copy, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = self->to(* other, non_blocking, copy, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Scalar> cpp_torch_method_item_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->item();
return make_xptr<torch::Scalar>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_set__self_Tensor_source_Storage (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Storage> source) {
  auto r_out = self->set_(* source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_set__self_Tensor_source_Storage_storage_offset_int64_t_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Storage> source, int64_t storage_offset, std::vector<int64_t> size, std::vector<int64_t> stride) {
  auto r_out = self->set_(* source, storage_offset, size, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_set__self_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->set_(* source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_set__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->set_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_is_set_to_self_Tensor_tensor_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor) {
  auto r_out = self->is_set_to(* tensor);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->masked_fill_(* mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->masked_fill(* mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_fill__self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->masked_fill_(* mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_fill_self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->masked_fill(* mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_scatter__self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->masked_scatter_(* mask, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->masked_scatter(* mask, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_view_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size) {
  auto r_out = self->view(size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_put__self_Tensor_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source, bool accumulate) {
  auto r_out = self->put_(* index, * source, accumulate);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_add__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_add_(dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_add(dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = self->index_add(* dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->index_fill_(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->index_fill(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill__self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->index_fill_(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->index_fill(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->index_fill_(* dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill__self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->index_fill_(* dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->index_fill(* dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = self->index_fill(* dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter_(dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter(dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter__self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->scatter_(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->scatter(dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter(* dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->scatter(* dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_add__self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter_add_(dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter_add(dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = self->scatter_add(* dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lt__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->lt_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lt__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->lt_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gt__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->gt_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gt__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->gt_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_le__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->le_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_le__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->le_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ge__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->ge_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ge__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->ge_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_eq__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->eq_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_eq__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->eq_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ne__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->ne_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ne__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->ne_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___and___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__and__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___and___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__and__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___iand___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__iand__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___iand___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__iand__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___or___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__or__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___or___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__or__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ior___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__ior__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ior___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__ior__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_xor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->bitwise_xor(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_xor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->bitwise_xor(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_xor__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->bitwise_xor_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_bitwise_xor__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->bitwise_xor_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___xor___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__xor__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___xor___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__xor__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ixor___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__ixor__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ixor___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__ixor__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___lshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__lshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___lshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__lshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ilshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__ilshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___ilshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__ilshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___rshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__rshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___rshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__rshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___irshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->__irshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method___irshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->__irshift__(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lgamma__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->lgamma_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_atan2__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->atan2_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tril__self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = self->tril_(diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_triu__self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = self->triu_(diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_digamma__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->digamma_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_polygamma__self_Tensor_n_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n) {
  auto r_out = self->polygamma_(n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_renorm__self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, int64_t dim, Rcpp::XPtr<torch::Scalar> maxnorm) {
  auto r_out = self->renorm_(* p, dim, * maxnorm);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pow__self_Tensor_exponent_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> exponent) {
  auto r_out = self->pow_(* exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pow__self_Tensor_exponent_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = self->pow_(* exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Scalar> weight) {
  auto r_out = self->lerp_(* end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lerp__self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = self->lerp_(* end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fmod__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->fmod_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fmod__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->fmod_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_remainder__self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->remainder_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_remainder__self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->remainder_(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addbmm__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addbmm_(* batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = self->addbmm(* batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addcdiv__self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->addcdiv_(* tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_random__self_Tensor_from_int64_t_to_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t from, int64_t to, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->random_(from, to, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_random__self_Tensor_to_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t to, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->random_(to, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_random__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->random_(* generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_uniform__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double from, double to, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->uniform_(from, to, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_normal__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double mean, double std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->normal_(mean, std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cauchy__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double median, double sigma, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->cauchy_(median, sigma, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_log_normal__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double mean, double std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->log_normal_(mean, std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_exponential__self_Tensor (Rcpp::XPtr<torch::Tensor> self, double lambd, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->exponential_(lambd, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_geometric__self_Tensor_p_double (Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->geometric_(p, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_diag_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = self->diag(diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cross_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, int64_t dim) {
  auto r_out = self->cross(* other, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_triu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = self->triu(diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_tril_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = self->tril(diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_trace_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->trace();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ne_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->ne(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ne_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->ne(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_eq_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->eq(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_eq_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->eq(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ge_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->ge(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ge_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->ge(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_le_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->le(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_le_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->le(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gt_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->gt(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gt_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->gt(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lt_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->lt(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lt_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->lt(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_take_self_Tensor_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = self->take(* index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_select_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = self->index_select(dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_index_select_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = self->index_select(* dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_masked_select_self_Tensor_mask_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask) {
  auto r_out = self->masked_select(* mask);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_nonzero_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->nonzero();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_method_nonzero_numpy_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->nonzero_numpy();
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gather_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = self->gather(dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_gather_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = self->gather(* dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->addcmul(* tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addcmul__self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->addcmul_(* tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = self->addcdiv(* tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_lstsq_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = self->lstsq(* A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_triangular_solve_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = self->triangular_solve(* A, upper, transpose, unitriangular);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_symeig_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool eigenvectors, bool upper) {
  auto r_out = self->symeig(eigenvectors, upper);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_eig_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool eigenvectors) {
  auto r_out = self->eig(eigenvectors);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_svd_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool some, bool compute_uv) {
  auto r_out = self->svd(some, compute_uv);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cholesky_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = self->cholesky(upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cholesky_solve_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, bool upper) {
  auto r_out = self->cholesky_solve(* input2, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_solve_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = self->solve(* A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_cholesky_inverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = self->cholesky_inverse(upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_qr_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool some) {
  auto r_out = self->qr(some);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_geqrf_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->geqrf();
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_orgqr_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2) {
  auto r_out = self->orgqr(* input2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_ormqr_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> input3, bool left, bool transpose) {
  auto r_out = self->ormqr(* input2, * input3, left, transpose);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> LU_data, Rcpp::XPtr<torch::Tensor> LU_pivots) {
  auto r_out = self->lu_solve(* LU_data, * LU_pivots);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_multinomial_self_Tensor_num_samples_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t num_samples, bool replacement, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = self->multinomial(num_samples, replacement, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lgamma_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->lgamma();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_digamma_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->digamma();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_polygamma_n_int64_t_self_Tensor (int64_t n, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->polygamma(n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erfinv_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erfinv();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_erfinv__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->erfinv_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sign_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sign();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_sign__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->sign_();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_dist_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> p) {
  auto r_out = self->dist(* other, * p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_atan2_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->atan2(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Scalar> weight) {
  auto r_out = self->lerp(* end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_lerp_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = self->lerp(* end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_histc_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t bins, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = self->histc(bins, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fmod_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->fmod(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_fmod_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->fmod(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_remainder_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = self->remainder(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_remainder_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->remainder(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_min_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->min(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_min_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->min();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_max_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->max(* other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_max_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->max();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_median_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->median();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool descending) {
  auto r_out = self->sort(dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_sort_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool descending) {
  auto r_out = self->sort(* dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_argsort_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool descending) {
  auto r_out = self->argsort(dim, descending);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_argsort_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool descending) {
  auto r_out = self->argsort(* dim, descending);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_method_topk_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool largest, bool sorted) {
  auto r_out = self->topk(k, dim, largest, sorted);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_all_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->all();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_any_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->any();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, int64_t dim, Rcpp::XPtr<torch::Scalar> maxnorm) {
  auto r_out = self->renorm(* p, dim, * maxnorm);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_unfold_self_Tensor_dimension_int64_t_size_int64_t_step_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dimension, int64_t size, int64_t step) {
  auto r_out = self->unfold(dimension, size, step);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_method_equal_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = self->equal(* other);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_pow_self_Tensor_exponent_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = self->pow(* exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_method_alias_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = self->alias();
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Byte_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Byte(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Char_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Char(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Double_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Double(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Float_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Float(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Int_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Int(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Long_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Long(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Short_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Short(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cast_Half_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool non_blocking) {
  auto r_out = torch::_cast_Half(* self, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_align_tensors_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors) {
  auto r_out = torch::align_tensors(* tensors);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace__use_cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t (Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, int64_t blank) {
  auto r_out = torch::_use_cudnn_ctc_loss(* log_probs, * targets, input_lengths, target_lengths, blank);
return r_out;
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_blank_int64_t_deterministic_bool_zero_infinity_bool (Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  auto r_out = torch::_cudnn_ctc_loss(* log_probs, * targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cudnn_rnn_flatten_weight_weight_arr_TensorList_weight_stride0_int64_t_input_size_int64_t_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_bidirectional_bool (Rcpp::XPtr<std::vector<torch::Tensor>> weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  auto r_out = torch::_cudnn_rnn_flatten_weight(* weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> weight, int64_t weight_stride0, Rcpp::XPtr<torch::Tensor> weight_buf, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<torch::Tensor> dropout_state) {
  auto r_out = torch::_cudnn_rnn(* input, * weight, weight_stride0, * weight_buf, * hx, * cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, * dropout_state);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__cudnn_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> weight, int64_t weight_stride0, Rcpp::XPtr<torch::Tensor> weight_buf, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> cx, Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<torch::Tensor> dropout_state, Rcpp::XPtr<torch::Tensor> reserve, std::vector<bool> output_mask) {
  auto r_out = torch::_cudnn_rnn_backward(* input, * weight, weight_stride0, * weight_buf, * hx, * cx, * output, * grad_output, * grad_hy, * grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, * dropout_state, * reserve, std_vector_to_std_array<bool,4>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::TensorList>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cudnn_init_dropout_state_dropout_double_train_bool_dropout_seed_int64_t_options_TensorOptions (double dropout, bool train, int64_t dropout_seed, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::_cudnn_init_dropout_state(dropout, train, dropout_seed, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__debug_has_internal_overlap_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_debug_has_internal_overlap(* self);
return r_out;
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__fused_dropout_self_Tensor_p_double (Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::_fused_dropout(* self, p, * generator);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__masked_scale_self_Tensor_mask_Tensor_scale_double (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, double scale) {
  auto r_out = torch::_masked_scale(* self, * mask, scale);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__sobol_engine_draw_quasi_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> quasi, int64_t n, Rcpp::XPtr<torch::Tensor> sobolstate, int64_t dimension, int64_t num_generated, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::_sobol_engine_draw(* quasi, n, * sobolstate, dimension, num_generated, * dtype);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sobol_engine_ff__self_Tensor_n_int64_t_sobolstate_Tensor_dimension_int64_t_num_generated_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n, Rcpp::XPtr<torch::Tensor> sobolstate, int64_t dimension, int64_t num_generated) {
  auto r_out = torch::_sobol_engine_ff_(* self, n, * sobolstate, dimension, num_generated);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sobol_engine_scramble__self_Tensor_ltm_Tensor_dimension_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> ltm, int64_t dimension) {
  auto r_out = torch::_sobol_engine_scramble_(* self, * ltm, dimension);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sobol_engine_initialize_state__self_Tensor_dimension_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dimension) {
  auto r_out = torch::_sobol_engine_initialize_state_(* self, dimension);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__reshape_from_tensor_self_Tensor_shape_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> shape) {
  auto r_out = torch::_reshape_from_tensor(* self, * shape);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__shape_as_tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_shape_as_tensor(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> input, double p, bool train) {
  auto r_out = torch::dropout(* input, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> self, double p, bool train) {
  auto r_out = torch::dropout_(* self, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_feature_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> input, double p, bool train) {
  auto r_out = torch::feature_dropout(* input, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_feature_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> self, double p, bool train) {
  auto r_out = torch::feature_dropout_(* self, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_alpha_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> input, double p, bool train) {
  auto r_out = torch::alpha_dropout(* input, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_alpha_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> self, double p, bool train) {
  auto r_out = torch::alpha_dropout_(* self, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_feature_alpha_dropout_input_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> input, double p, bool train) {
  auto r_out = torch::feature_alpha_dropout(* input, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_feature_alpha_dropout__self_Tensor_p_double_train_bool (Rcpp::XPtr<torch::Tensor> self, double p, bool train) {
  auto r_out = torch::feature_alpha_dropout_(* self, p, train);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_abs_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::abs(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_abs__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::abs_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_abs_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::abs_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_angle_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::angle(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_angle_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::angle_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_real_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::real(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_real_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::real_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_imag_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::imag(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_imag_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::imag_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conj_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::conj(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conj_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::conj_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_acos_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::acos(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_acos__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::acos_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_acos_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::acos_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool1d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad) {
  auto r_out = torch::avg_pool1d(* self, kernel_size, stride, padding, ceil_mode, count_include_pad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_avg_pool1d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_max_pool1d(* self, output_size);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_add_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::add(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_add_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::add_out(* out, * self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_add_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::add(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addmv_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat, Rcpp::XPtr<torch::Tensor> vec, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addmv(* self, * mat, * vec, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addmv__self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat, Rcpp::XPtr<torch::Tensor> vec, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addmv_(* self, * mat, * vec, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addmv_out_out_Tensor_self_Tensor_mat_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat, Rcpp::XPtr<torch::Tensor> vec, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addmv_out(* out, * self, * mat, * vec, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addr_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addr(* self, * vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addr_out_out_Tensor_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addr_out(* out, * self, * vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_affine_grid_generator_theta_Tensor_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> theta, std::vector<int64_t> size, bool align_corners) {
  auto r_out = torch::affine_grid_generator(* theta, size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_affine_grid_generator_backward_grad_Tensor_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad, std::vector<int64_t> size, bool align_corners) {
  auto r_out = torch::affine_grid_generator_backward(* grad, size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_all_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::all(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::all_out(* out, * self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_all_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::all(* self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_all_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::all_out(* out, * self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_allclose_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = torch::allclose(* self, * other, rtol, atol, equal_nan);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_any_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::any(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::any_out(* out, * self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_any_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::any(* self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_any_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::any_out(* out, * self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_arange_end_Scalar (Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::arange(* end, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_arange_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::arange(* start, * end, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_arange_start_Scalar_end_Scalar_step_Scalar (Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::Scalar> step, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::arange(* start, * end, * step, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_arange_out_out_Tensor_end_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> end) {
  auto r_out = torch::arange_out(* out, * end);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_arange_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::Scalar> step) {
  auto r_out = torch::arange_out(* out, * start, * end, * step);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__dim_arange_like_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> like, int64_t dim) {
  auto r_out = torch::_dim_arange(* like, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_argmax_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::argmax(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_argmin_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::argmin(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_as_strided_self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, int64_t storage_offset) {
  auto r_out = torch::as_strided(* self, size, stride, storage_offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_as_strided__self_Tensor_size_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size, std::vector<int64_t> stride, int64_t storage_offset) {
  auto r_out = torch::as_strided_(* self, size, stride, storage_offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_asin_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::asin(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_asin__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::asin_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_asin_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::asin_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_atan_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::atan(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_atan__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::atan_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_atan_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::atan_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_baddbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::baddbmm(* self, * batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__baddbmm_mkl__self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::_baddbmm_mkl_(* self, * batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_baddbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::baddbmm_out(* out, * self, * batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bartlett_window_window_length_int64_t (int64_t window_length, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::bartlett_window(window_length, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bartlett_window_window_length_int64_t_periodic_bool (int64_t window_length, bool periodic, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::bartlett_window(window_length, periodic, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = torch::batch_norm(* input, * weight, * bias, * running_mean, * running_var, training, momentum, eps, cudnn_enabled);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = torch::_batch_norm_impl_index(* input, * weight, * bias, * running_mean, * running_var, training, momentum, eps, cudnn_enabled);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),std::get<4>(r_out));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__batch_norm_impl_index_backward_impl_index_int64_t_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_transform_Tensor_train_bool_eps_double_output_mask_stdarraybool3_reservedSpace_Tensor (int64_t impl_index, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, Rcpp::XPtr<torch::Tensor> save_mean, Rcpp::XPtr<torch::Tensor> save_var_transform, bool train, double eps, std::vector<bool> output_mask, Rcpp::XPtr<torch::Tensor> reservedSpace) {
  auto r_out = torch::_batch_norm_impl_index_backward(impl_index, * input, * grad_output, * weight, * running_mean, * running_var, * save_mean, * save_var_transform, train, eps, std_vector_to_std_array<bool,3>(output_mask), * reservedSpace);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bernoulli_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::bernoulli(* self, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bernoulli_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::bernoulli_out(* out, * self, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bernoulli_self_Tensor_p_double (Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::bernoulli(* self, p, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bilinear_input1_Tensor_input2_Tensor_weight_Tensor_bias_Tensor (Rcpp::XPtr<torch::Tensor> input1, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::bilinear(* input1, * input2, * weight, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_with_logits_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> pos_weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy_with_logits(* self, * target, * weight, * pos_weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_with_logits_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> pos_weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy_with_logits_backward(* grad_output, * self, * target, * weight, * pos_weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bincount_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weights, int64_t minlength) {
  auto r_out = torch::bincount(* self, * weights, minlength);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_not_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::bitwise_not(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_not_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::bitwise_not_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logical_not_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::logical_not(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logical_not_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::logical_not_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logical_xor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::logical_xor(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logical_xor_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::logical_xor_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_blackman_window_window_length_int64_t (int64_t window_length, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::blackman_window(window_length, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_blackman_window_window_length_int64_t_periodic_bool (int64_t window_length, bool periodic, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::blackman_window(window_length, periodic, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bmm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::bmm(* self, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bmm_out_out_Tensor_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::bmm_out(* out, * self, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_broadcast_tensors_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors) {
  auto r_out = torch::broadcast_tensors(* tensors);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cat_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::cat(* tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::cat_out(* out, * tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cat_tensors_TensorList_dim_Dimname (Rcpp::XPtr<std::vector<torch::Tensor>> tensors, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::cat(* tensors, * dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cat_out_out_Tensor_tensors_TensorList_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<std::vector<torch::Tensor>> tensors, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::cat_out(* out, * tensors, * dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ceil_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::ceil(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ceil__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::ceil_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ceil_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::ceil_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_chain_matmul_matrices_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> matrices) {
  auto r_out = torch::chain_matmul(* matrices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_chunk_self_Tensor_chunks_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t chunks, int64_t dim) {
  auto r_out = torch::chunk(* self, chunks, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp(* self, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp_(* self, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp_out(* out, * self, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_max_self_Tensor_max_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp_max(* self, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_max__self_Tensor_max_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp_max_(* self, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_max_out_out_Tensor_self_Tensor_max_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::clamp_max_out(* out, * self, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_min_self_Tensor_min_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min) {
  auto r_out = torch::clamp_min(* self, * min);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_min__self_Tensor_min_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min) {
  auto r_out = torch::clamp_min_(* self, * min);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clamp_min_out_out_Tensor_self_Tensor_min_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min) {
  auto r_out = torch::clamp_min_out(* out, * self, * min);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_cudnn_is_acceptable_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cudnn_is_acceptable(* self);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_constant_pad_nd_self_Tensor_pad_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> pad, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::constant_pad_nd(* self, pad, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups) {
  auto r_out = torch::convolution(* input, * weight, * bias, stride, padding, dilation, transposed, output_padding, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_convolution_overrideable_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups) {
  auto r_out = torch::convolution_overrideable(* input, * weight, * bias, stride, padding, dilation, transposed, output_padding, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_convolution_backward_overrideable_grad_output_Tensor_input_Tensor_weight_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups, std::vector<bool> output_mask) {
  auto r_out = torch::convolution_backward_overrideable(* grad_output, * input, * weight, stride, padding, dilation, transposed, output_padding, groups, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__convolution_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  auto r_out = torch::_convolution(* input, * weight, * bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__convolution_nogroup_input_Tensor_weight_Tensor_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding) {
  auto r_out = torch::_convolution_nogroup(* input, * weight, * bias, stride, padding, dilation, transposed, output_padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__convolution_double_backward_ggI_Tensor_ggW_Tensor_ggb_Tensor_gO_Tensor_weight_Tensor_self_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_transposed_bool_output_padding_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_cudnn_enabled_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> ggI, Rcpp::XPtr<torch::Tensor> ggW, Rcpp::XPtr<torch::Tensor> ggb, Rcpp::XPtr<torch::Tensor> gO, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::vector<bool> output_mask) {
  auto r_out = torch::_convolution_double_backward(* ggI, * ggW, * ggb, * gO, * weight, * self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv1d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t groups) {
  auto r_out = torch::conv1d(* input, * weight, * bias, stride, padding, dilation, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv2d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t groups) {
  auto r_out = torch::conv2d(* input, * weight, * bias, stride, padding, dilation, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv3d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t groups) {
  auto r_out = torch::conv3d(* input, * weight, * bias, stride, padding, dilation, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv_tbc_self_Tensor_weight_Tensor_bias_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, int64_t pad) {
  auto r_out = torch::conv_tbc(* self, * weight, * bias, pad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_conv_tbc_backward_self_Tensor_input_Tensor_weight_Tensor_bias_Tensor_pad_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, int64_t pad) {
  auto r_out = torch::conv_tbc_backward(* self, * input, * weight, * bias, pad);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv_transpose1d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, int64_t groups, std::vector<int64_t> dilation) {
  auto r_out = torch::conv_transpose1d(* input, * weight, * bias, stride, padding, output_padding, groups, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv_transpose2d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, int64_t groups, std::vector<int64_t> dilation) {
  auto r_out = torch::conv_transpose2d(* input, * weight, * bias, stride, padding, output_padding, groups, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_conv_transpose3d_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, int64_t groups, std::vector<int64_t> dilation) {
  auto r_out = torch::conv_transpose3d(* input, * weight, * bias, stride, padding, output_padding, groups, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__copy_from_self_Tensor_dst_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> dst, bool non_blocking) {
  auto r_out = torch::_copy_from(* self, * dst, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cos_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cos(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cos__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cos_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cos_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cos_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cosh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cosh(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cosh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cosh_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cosh_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::cosh_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cosine_embedding_loss_input1_Tensor_input2_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> input1, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> target, double margin, int64_t reduction) {
  auto r_out = torch::cosine_embedding_loss(* input1, * input2, * target, margin, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_affine_grid_generator_theta_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (Rcpp::XPtr<torch::Tensor> theta, int64_t False, int64_t C, int64_t H, int64_t W) {
  auto r_out = torch::cudnn_affine_grid_generator(* theta, False, C, H, W);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_affine_grid_generator_backward_grad_Tensor_FALSE_int64_t_C_int64_t_H_int64_t_W_int64_t (Rcpp::XPtr<torch::Tensor> grad, int64_t False, int64_t C, int64_t H, int64_t W) {
  auto r_out = torch::cudnn_affine_grid_generator_backward(* grad, False, C, H, W);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool training, double exponential_average_factor, double epsilon) {
  auto r_out = torch::cudnn_batch_norm(* input, * weight, * bias, * running_mean, * running_var, training, exponential_average_factor, epsilon);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double_reserveSpace_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, Rcpp::XPtr<torch::Tensor> save_mean, Rcpp::XPtr<torch::Tensor> save_var, double epsilon, Rcpp::XPtr<torch::Tensor> reserveSpace) {
  auto r_out = torch::cudnn_batch_norm_backward(* input, * grad_output, * weight, * running_mean, * running_var, * save_mean, * save_var, epsilon, * reserveSpace);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution(* self, * weight, * bias, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> self_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution_backward_input(self_size, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = torch::cudnn_convolution_backward(* self, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_backward_bias_grad_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output) {
  auto r_out = torch::cudnn_convolution_backward_bias(* grad_output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution_backward_weight(weight_size, * grad_output, * self, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution_transpose(* self, * weight, * bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = torch::cudnn_convolution_transpose_backward(* self, * grad_output, * weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_transpose_backward_bias_grad_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output) {
  auto r_out = torch::cudnn_convolution_transpose_backward_bias(* grad_output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution_transpose_backward_input(* grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::cudnn_convolution_transpose_backward_weight(weight_size, * grad_output, * self, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cudnn_grid_sampler_self_Tensor_grid_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grid) {
  auto r_out = torch::cudnn_grid_sampler(* self, * grid);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_cudnn_grid_sampler_backward_self_Tensor_grid_Tensor_grad_output_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grid, Rcpp::XPtr<torch::Tensor> grad_output) {
  auto r_out = torch::cudnn_grid_sampler_backward(* self, * grid, * grad_output);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumsum(* self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumsum_out(* out, * self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumsum_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumsum(* self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumsum_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumsum_out(* out, * self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumprod(* self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumprod_out(* out, * self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumprod_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumprod(* self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cumprod_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::cumprod_out(* out, * self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  auto r_out = torch::ctc_loss(* log_probs, * targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_Tensor_target_lengths_Tensor (Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, Rcpp::XPtr<torch::Tensor> input_lengths, Rcpp::XPtr<torch::Tensor> target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  auto r_out = torch::ctc_loss(* log_probs, * targets, * input_lengths, * target_lengths, blank, reduction, zero_infinity);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__ctc_loss_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef (Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, int64_t blank, bool zero_infinity) {
  auto r_out = torch::_ctc_loss(* log_probs, * targets, input_lengths, target_lengths, blank, zero_infinity);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__ctc_loss_backward_grad_Tensor_log_probs_Tensor_targets_Tensor_input_lengths_IntArrayRef_target_lengths_IntArrayRef_neg_log_likelihood_Tensor_log_alpha_Tensor_blank_int64_t (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> log_probs, Rcpp::XPtr<torch::Tensor> targets, std::vector<int64_t> input_lengths, std::vector<int64_t> target_lengths, Rcpp::XPtr<torch::Tensor> neg_log_likelihood, Rcpp::XPtr<torch::Tensor> log_alpha, int64_t blank, bool zero_infinity) {
  auto r_out = torch::_ctc_loss_backward(* grad, * log_probs, * targets, input_lengths, target_lengths, * neg_log_likelihood, * log_alpha, blank, zero_infinity);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_det_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::det(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_diag_embed_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto r_out = torch::diag_embed(* self, offset, dim1, dim2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_diagflat_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset) {
  auto r_out = torch::diagflat(* self, offset);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_diagonal_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto r_out = torch::diagonal(* self, offset, dim1, dim2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_div_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::div(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_div_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::div_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_div_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::div(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dot_self_Tensor_tensor_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor) {
  auto r_out = torch::dot(* self, * tensor);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dot_out_out_Tensor_self_Tensor_tensor_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor) {
  auto r_out = torch::dot_out(* out, * self, * tensor);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_einsum_equation_stdstring_tensors_TensorList (std::string equation, Rcpp::XPtr<std::vector<torch::Tensor>> tensors) {
  auto r_out = torch::einsum(equation, * tensors);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_embedding_weight_Tensor_indices_Tensor (Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto r_out = torch::embedding(* weight, * indices, padding_idx, scale_grad_by_freq, sparse);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_embedding_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool_sparse_bool (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto r_out = torch::embedding_backward(* grad, * indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_embedding_dense_backward_grad_output_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  auto r_out = torch::embedding_dense_backward(* grad_output, * indices, num_weights, padding_idx, scale_grad_by_freq);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_embedding_renorm__self_Tensor_indices_Tensor_max_norm_double_norm_type_double (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, double max_norm, double norm_type) {
  auto r_out = torch::embedding_renorm_(* self, * indices, max_norm, norm_type);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_embedding_sparse_backward_grad_Tensor_indices_Tensor_num_weights_int64_t_padding_idx_int64_t_scale_grad_by_freq_bool (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  auto r_out = torch::embedding_sparse_backward(* grad, * indices, num_weights, padding_idx, scale_grad_by_freq);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor (Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, Rcpp::XPtr<torch::Tensor> per_sample_weights) {
  auto r_out = torch::embedding_bag(* weight, * indices, * offsets, scale_grad_by_freq, mode, sparse, * per_sample_weights);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__embedding_bag_weight_Tensor_indices_Tensor_offsets_Tensor (Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, Rcpp::XPtr<torch::Tensor> per_sample_weights) {
  auto r_out = torch::_embedding_bag(* weight, * indices, * offsets, scale_grad_by_freq, mode, sparse, * per_sample_weights);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__embedding_bag_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_sparse_bool_per_sample_weights_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, Rcpp::XPtr<torch::Tensor> offset2bag, Rcpp::XPtr<torch::Tensor> bag_size, Rcpp::XPtr<torch::Tensor> maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, Rcpp::XPtr<torch::Tensor> per_sample_weights) {
  auto r_out = torch::_embedding_bag_backward(* grad, * indices, * offsets, * offset2bag, * bag_size, * maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, * per_sample_weights);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__embedding_bag_sparse_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, Rcpp::XPtr<torch::Tensor> offset2bag, Rcpp::XPtr<torch::Tensor> bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, Rcpp::XPtr<torch::Tensor> per_sample_weights) {
  auto r_out = torch::_embedding_bag_sparse_backward(* grad, * indices, * offsets, * offset2bag, * bag_size, num_weights, scale_grad_by_freq, mode, * per_sample_weights);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__embedding_bag_dense_backward_grad_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_bag_size_Tensor_maximum_indices_Tensor_num_weights_int64_t_scale_grad_by_freq_bool_mode_int64_t_per_sample_weights_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, Rcpp::XPtr<torch::Tensor> offset2bag, Rcpp::XPtr<torch::Tensor> bag_size, Rcpp::XPtr<torch::Tensor> maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, Rcpp::XPtr<torch::Tensor> per_sample_weights) {
  auto r_out = torch::_embedding_bag_dense_backward(* grad, * indices, * offsets, * offset2bag, * bag_size, * maximum_indices, num_weights, scale_grad_by_freq, mode, * per_sample_weights);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__embedding_bag_per_sample_weights_backward_grad_Tensor_weight_Tensor_indices_Tensor_offsets_Tensor_offset2bag_Tensor_mode_int64_t (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> offsets, Rcpp::XPtr<torch::Tensor> offset2bag, int64_t mode) {
  auto r_out = torch::_embedding_bag_per_sample_weights_backward(* grad, * weight, * indices, * offsets, * offset2bag, mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::empty(size, * names, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::empty(size, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__empty_affine_quantized_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options, double scale, int64_t zero_point, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::_empty_affine_quantized(size, * options, scale, zero_point, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__empty_per_channel_affine_quantized_size_IntArrayRef_scales_Tensor_zero_points_Tensor_axis_int64_t (std::vector<int64_t> size, Rcpp::XPtr<torch::Tensor> scales, Rcpp::XPtr<torch::Tensor> zero_points, int64_t axis, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::_empty_per_channel_affine_quantized(size, * scales, * zero_points, axis, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::empty_out(* out, size, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_like_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::empty_like(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_like_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::empty_like(* self, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_empty_strided_size_IntArrayRef_stride_IntArrayRef (std::vector<int64_t> size, std::vector<int64_t> stride, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::empty_strided(size, stride, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erf_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erf(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erf__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erf_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erf_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erf_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erfc_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erfc(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erfc__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erfc_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erfc_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erfc_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_exp_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::exp(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_exp__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::exp_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_exp_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::exp_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_expm1_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::expm1(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_expm1__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::expm1_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_expm1_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::expm1_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eye_n_int64_t (int64_t n, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::eye(n, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eye_n_int64_t_m_int64_t (int64_t n, int64_t m, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::eye(n, m, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eye_out_out_Tensor_n_int64_t (Rcpp::XPtr<torch::Tensor> out, int64_t n) {
  auto r_out = torch::eye_out(* out, n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eye_out_out_Tensor_n_int64_t_m_int64_t (Rcpp::XPtr<torch::Tensor> out, int64_t n, int64_t m) {
  auto r_out = torch::eye_out(* out, n, m);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_flatten_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t start_dim, int64_t end_dim) {
  auto r_out = torch::flatten(* self, start_dim, end_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_flatten_self_Tensor_start_dim_int64_t_end_dim_int64_t_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, int64_t start_dim, int64_t end_dim, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = torch::flatten(* self, start_dim, end_dim, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_flatten_self_Tensor_start_dim_Dimname_end_dim_Dimname_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> start_dim, Rcpp::XPtr<torch::Dimname> end_dim, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = torch::flatten(* self, * start_dim, * end_dim, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_flatten_self_Tensor_dims_DimnameList_out_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dims, Rcpp::XPtr<torch::Dimname> out_dim) {
  auto r_out = torch::flatten(* self, * dims, * out_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fill__self_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::fill_(* self, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fill__self_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = torch::fill_(* self, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_floor_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::floor(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_floor__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::floor_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_floor_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::floor_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frac_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::frac(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frac__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::frac_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frac_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::frac_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<torch::Scalar> fill_value, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::full(size, * fill_value, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_full_size_IntArrayRef_fill_value_Scalar (std::vector<int64_t> size, Rcpp::XPtr<torch::Scalar> fill_value, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::full(size, * fill_value, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_full_out_out_Tensor_size_IntArrayRef_fill_value_Scalar (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size, Rcpp::XPtr<torch::Scalar> fill_value) {
  auto r_out = torch::full_out(* out, size, * fill_value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_full_like_self_Tensor_fill_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> fill_value, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::full_like(* self, * fill_value, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_full_like_self_Tensor_fill_value_Scalar_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> fill_value, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::full_like(* self, * fill_value, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_from_file_filename_stdstring (std::string filename, bool shared, int64_t size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::from_file(filename, shared, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_grid_sampler_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto r_out = torch::grid_sampler(* input, * grid, interpolation_mode, padding_mode, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_grid_sampler_2d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto r_out = torch::grid_sampler_2d(* input, * grid, interpolation_mode, padding_mode, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_2d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto r_out = torch::grid_sampler_2d_backward(* grad_output, * input, * grid, interpolation_mode, padding_mode, align_corners);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_grid_sampler_3d_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto r_out = torch::grid_sampler_3d(* input, * grid, interpolation_mode, padding_mode, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_grid_sampler_3d_backward_grad_output_Tensor_input_Tensor_grid_Tensor_interpolation_mode_int64_t_padding_mode_int64_t_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto r_out = torch::grid_sampler_3d_backward(* grad_output, * input, * grid, interpolation_mode, padding_mode, align_corners);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hann_window_window_length_int64_t (int64_t window_length, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hann_window(window_length, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hann_window_window_length_int64_t_periodic_bool (int64_t window_length, bool periodic, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hann_window(window_length, periodic, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hamming_window_window_length_int64_t (int64_t window_length, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hamming_window(window_length, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool (int64_t window_length, bool periodic, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hamming_window(window_length, periodic, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double (int64_t window_length, bool periodic, double alpha, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hamming_window(window_length, periodic, alpha, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hamming_window_window_length_int64_t_periodic_bool_alpha_double_beta_double (int64_t window_length, bool periodic, double alpha, double beta, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::hamming_window(window_length, periodic, alpha, beta, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hinge_embedding_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, double margin, int64_t reduction) {
  auto r_out = torch::hinge_embedding_loss(* self, * target, margin, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ger_self_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec2) {
  auto r_out = torch::ger(* self, * vec2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ger_out_out_Tensor_self_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec2) {
  auto r_out = torch::ger_out(* out, * self, * vec2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_group_norm_input_Tensor_num_groups_int64_t (Rcpp::XPtr<torch::Tensor> input, int64_t num_groups, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, double eps, bool cudnn_enabled) {
  auto r_out = torch::group_norm(* input, num_groups, * weight, * bias, eps, cudnn_enabled);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized) {
  auto r_out = torch::fft(* self, signal_ndim, normalized);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ifft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized) {
  auto r_out = torch::ifft(* self, signal_ndim, normalized);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized, bool onesided) {
  auto r_out = torch::rfft(* self, signal_ndim, normalized, onesided);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_irfft_self_Tensor_signal_ndim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool normalized, bool onesided, std::vector<int64_t> signal_sizes) {
  auto r_out = torch::irfft(* self, signal_ndim, normalized, onesided, signal_sizes);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__fft_with_size_self_Tensor_signal_ndim_int64_t_complex_input_bool_complex_output_bool_inverse_bool_checked_signal_sizes_IntArrayRef_normalized_bool_onesided_bool_output_sizes_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, std::vector<int64_t> checked_signal_sizes, bool normalized, bool onesided, std::vector<int64_t> output_sizes) {
  auto r_out = torch::_fft_with_size(* self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__cufft_get_plan_cache_size_device_index_int64_t (int64_t device_index) {
  auto r_out = torch::_cufft_get_plan_cache_size(device_index);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace__cufft_get_plan_cache_max_size_device_index_int64_t (int64_t device_index) {
  auto r_out = torch::_cufft_get_plan_cache_max_size(device_index);
return r_out;
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_set_plan_cache_max_size_device_index_int64_t_max_size_int64_t (int64_t device_index, int64_t max_size) {
  torch::_cufft_set_plan_cache_max_size(device_index, max_size);
}

// [[Rcpp::export]]
void cpp_torch_namespace__cufft_clear_plan_cache_device_index_int64_t (int64_t device_index) {
  torch::_cufft_clear_plan_cache(device_index);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_self_Tensor_indices_TensorList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices) {
  auto r_out = torch::index(* self, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_copy_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::index_copy(* self, dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_copy_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::index_copy(* self, * dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_put__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices, Rcpp::XPtr<torch::Tensor> values, bool accumulate) {
  auto r_out = torch::index_put_(* self, * indices, * values, accumulate);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_put_self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices, Rcpp::XPtr<torch::Tensor> values, bool accumulate) {
  auto r_out = torch::index_put(* self, * indices, * values, accumulate);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__index_put_impl__self_Tensor_indices_TensorList_values_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Tensor>> indices, Rcpp::XPtr<torch::Tensor> values, bool accumulate, bool unsafe) {
  auto r_out = torch::_index_put_impl_(* self, * indices, * values, accumulate, unsafe);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_instance_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_use_input_stats_bool_momentum_double_eps_double_cudnn_enabled_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  auto r_out = torch::instance_norm(* input, * weight, * bias, * running_mean, * running_var, use_input_stats, momentum, eps, cudnn_enabled);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_inverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::inverse(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_inverse_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::inverse_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__inverse_helper_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_inverse_helper(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_isclose_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, double rtol, double atol, bool equal_nan) {
  auto r_out = torch::isclose(* self, * other, rtol, atol, equal_nan);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_isnan_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::isnan(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_distributed_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::is_distributed(* self);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_floating_point_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::is_floating_point(* self);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_complex_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::is_complex(* self);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_nonzero_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::is_nonzero(* self);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_same_size_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::is_same_size(* self, * other);
return r_out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace_is_signed_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::is_signed(* self);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_kl_div_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::kl_div(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_kl_div_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::kl_div_backward(* grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool keepdim) {
  auto r_out = torch::kthvalue(* self, k, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool keepdim) {
  auto r_out = torch::kthvalue_out(* values, * indices, * self, k, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, int64_t k, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::kthvalue(* self, k, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_kthvalue_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t_dim_Dimname (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t k, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::kthvalue_out(* values, * indices, * self, k, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_layer_norm_input_Tensor_normalized_shape_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, std::vector<int64_t> normalized_shape, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, double eps, bool cudnn_enable) {
  auto r_out = torch::layer_norm(* input, normalized_shape, * weight, * bias, eps, cudnn_enable);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_input_Tensor_weight_Tensor_bias_Tensor_M_int64_t_FALSE_int64_t_eps_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, int64_t M, int64_t False, double eps) {
  auto r_out = torch::native_layer_norm(* input, * weight, * bias, M, False, eps);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_layer_norm_backward_grad_out_Tensor_input_Tensor_mean_Tensor_rstd_Tensor_weight_Tensor_M_int64_t_FALSE_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> rstd, Rcpp::XPtr<torch::Tensor> weight, int64_t M, int64_t False, std::vector<bool> output_mask) {
  auto r_out = torch::native_layer_norm_backward(* grad_out, * input, * mean, * rstd, * weight, M, False, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_linear_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::linear(* input, * weight, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_linear_input_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::mkldnn_linear(* input, * weight, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_linear_int8_weight_fp32_activation_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> packed, Rcpp::XPtr<torch::Tensor> col_offsets, Rcpp::XPtr<torch::Scalar> weight_scale, Rcpp::XPtr<torch::Scalar> weight_zero_point, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::fbgemm_linear_int8_weight_fp32_activation(* input, * weight, * packed, * col_offsets, * weight_scale, * weight_zero_point, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_linear_int8_weight_input_Tensor_weight_Tensor_packed_Tensor_col_offsets_Tensor_weight_scale_Scalar_weight_zero_point_Scalar_bias_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> packed, Rcpp::XPtr<torch::Tensor> col_offsets, Rcpp::XPtr<torch::Scalar> weight_scale, Rcpp::XPtr<torch::Scalar> weight_zero_point, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::fbgemm_linear_int8_weight(* input, * weight, * packed, * col_offsets, * weight_scale, * weight_zero_point, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fbgemm_linear_quantize_weight_input_Tensor (Rcpp::XPtr<torch::Tensor> input) {
  auto r_out = torch::fbgemm_linear_quantize_weight(* input);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),std::get<2>(r_out),std::get<3>(r_out));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_pack_gemm_matrix_fp16_input_Tensor (Rcpp::XPtr<torch::Tensor> input) {
  auto r_out = torch::fbgemm_pack_gemm_matrix_fp16(* input);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_linear_fp16_weight_fp32_activation_input_Tensor_packed_weight_Tensor_bias_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> packed_weight, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::fbgemm_linear_fp16_weight_fp32_activation(* input, * packed_weight, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_linear_fp16_weight_input_Tensor_packed_weight_Tensor_bias_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> packed_weight, Rcpp::XPtr<torch::Tensor> bias) {
  auto r_out = torch::fbgemm_linear_fp16_weight(* input, * packed_weight, * bias);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor (Rcpp::XPtr<torch::Tensor> input) {
  auto r_out = torch::fbgemm_pack_quantized_matrix(* input);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fbgemm_pack_quantized_matrix_input_Tensor_K_int64_t_FALSE_int64_t (Rcpp::XPtr<torch::Tensor> input, int64_t K, int64_t False) {
  auto r_out = torch::fbgemm_pack_quantized_matrix(* input, K, False);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_linspace_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, int64_t steps, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::linspace(* start, * end, steps, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_linspace_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, int64_t steps) {
  auto r_out = torch::linspace_out(* out, * start, * end, steps);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log10_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log10(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log10__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log10_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log10_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log10_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log1p_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log1p(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log1p__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log1p_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log1p_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log1p_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log2_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log2(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log2__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log2_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log2_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log2_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logdet_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::logdet(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logspace_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, int64_t steps, double base, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::logspace(* start, * end, steps, base, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logspace_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, int64_t steps, double base) {
  auto r_out = torch::logspace_out(* out, * start, * end, steps, base);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::log_softmax(* self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::log_softmax(* self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__log_softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool half_to_float) {
  auto r_out = torch::_log_softmax(* self, dim, half_to_float);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__log_softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output, int64_t dim, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_log_softmax_backward_data(* grad_output, * output, dim, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logsumexp_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::logsumexp(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::logsumexp_out(* out, * self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logsumexp_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::logsumexp(* self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_logsumexp_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::logsumexp_out(* out, * self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_margin_ranking_loss_input1_Tensor_input2_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> input1, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> target, double margin, int64_t reduction) {
  auto r_out = torch::margin_ranking_loss(* input1, * input2, * target, margin, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_matmul_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::matmul(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_matmul_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::matmul_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_matrix_rank_self_Tensor_tol_double (Rcpp::XPtr<torch::Tensor> self, double tol, bool symmetric) {
  auto r_out = torch::matrix_rank(* self, tol, symmetric);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_matrix_rank_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool symmetric) {
  auto r_out = torch::matrix_rank(* self, symmetric);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_matrix_power_self_Tensor_n_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n) {
  auto r_out = torch::matrix_power(* self, n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::max(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> max, Rcpp::XPtr<torch::Tensor> max_values, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::max_out(* max, * max_values, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_values_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::max_values(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::max(* self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_out_max_Tensor_max_values_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> max, Rcpp::XPtr<torch::Tensor> max_values, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::max_out(* max, * max_values, * self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_values_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::max_values(* self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool1d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool1d_with_indices(* self, kernel_size, stride, padding, dilation, ceil_mode);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool1d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool1d(* self, kernel_size, stride, padding, dilation, ceil_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool2d(* self, kernel_size, stride, padding, dilation, ceil_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::mkldnn_max_pool2d(* self, kernel_size, stride, padding, dilation, ceil_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantized_max_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::quantized_max_pool2d(* self, kernel_size, stride, padding, dilation, ceil_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool3d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool3d(* self, kernel_size, stride, padding, dilation, ceil_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mean_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::mean(* self, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::mean(* self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::mean_out(* out, * self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::mean(* self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mean_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::mean_out(* out, * self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::median(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::median_out(* values, * indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::median(* self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_median_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::median_out(* values, * indices, * self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::min(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> min, Rcpp::XPtr<torch::Tensor> min_indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::min_out(* min, * min_indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_min_values_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::min_values(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::min(* self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> min, Rcpp::XPtr<torch::Tensor> min_indices, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::min_out(* min, * min_indices, * self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_min_values_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::min_values(* self, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups) {
  auto r_out = torch::mkldnn_convolution(* self, * weight, * bias, padding, stride, dilation, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (std::vector<int64_t> self_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool bias_defined) {
  auto r_out = torch::mkldnn_convolution_backward_input(self_size, * grad_output, * weight, padding, stride, dilation, groups, bias_defined);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_weights_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_bias_defined_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool bias_defined) {
  auto r_out = torch::mkldnn_convolution_backward_weights(weight_size, * grad_output, * self, padding, stride, dilation, groups, bias_defined);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mkldnn_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, std::vector<bool> output_mask) {
  auto r_out = torch::mkldnn_convolution_backward(* self, * grad_output, * weight, padding, stride, dilation, groups, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_exponential_average_factor_double_epsilon_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool training, double exponential_average_factor, double epsilon) {
  auto r_out = torch::miopen_batch_norm(* input, * weight, * bias, * running_mean, * running_var, training, exponential_average_factor, epsilon);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_batch_norm_backward_input_Tensor_grad_output_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_var_Tensor_epsilon_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, Rcpp::XPtr<torch::Tensor> save_mean, Rcpp::XPtr<torch::Tensor> save_var, double epsilon) {
  auto r_out = torch::miopen_batch_norm_backward(* input, * grad_output, * weight, * running_mean, * running_var, * save_mean, * save_var, epsilon);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution(* self, * weight, * bias, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> self_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution_backward_input(self_size, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = torch::miopen_convolution_backward(* self, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_backward_bias_grad_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output) {
  auto r_out = torch::miopen_convolution_backward_bias(* grad_output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution_backward_weight(weight_size, * grad_output, * self, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_transpose_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution_transpose(* self, * weight, * bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_convolution_transpose_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = torch::miopen_convolution_transpose_backward(* self, * grad_output, * weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_transpose_backward_input_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution_transpose_backward_input(* grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_convolution_transpose_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_convolution_transpose_backward_weight(weight_size, * grad_output, * self, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_depthwise_convolution_self_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_depthwise_convolution(* self, * weight, * bias, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_depthwise_convolution_backward_input_self_size_IntArrayRef_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> self_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_depthwise_convolution_backward_input(self_size, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_depthwise_convolution_backward_self_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic, std::vector<bool> output_mask) {
  auto r_out = torch::miopen_depthwise_convolution_backward(* self, * grad_output, * weight, padding, stride, dilation, groups, benchmark, deterministic, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_miopen_depthwise_convolution_backward_weight_weight_size_IntArrayRef_grad_output_Tensor_self_Tensor_padding_IntArrayRef_stride_IntArrayRef_dilation_IntArrayRef_groups_int64_t_benchmark_bool_deterministic_bool (std::vector<int64_t> weight_size, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto r_out = torch::miopen_depthwise_convolution_backward_weight(weight_size, * grad_output, * self, padding, stride, dilation, groups, benchmark, deterministic);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_input_Tensor_weight_TensorList_weight_stride0_int64_t_hx_Tensor_cx_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> weight, int64_t weight_stride0, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<torch::Tensor> dropout_state) {
  auto r_out = torch::miopen_rnn(* input, * weight, weight_stride0, * hx, * cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, * dropout_state);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_miopen_rnn_backward_input_Tensor_weight_TensorList_weight_stride0_int64_t_weight_buf_Tensor_hx_Tensor_cx_Tensor_output_Tensor_grad_output_Tensor_grad_hy_Tensor_grad_cy_Tensor_mode_int64_t_hidden_size_int64_t_num_layers_int64_t_batch_first_bool_dropout_double_train_bool_bidirectional_bool_batch_sizes_IntArrayRef_dropout_state_Tensor_reserve_Tensor_output_mask_stdarraybool4 (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> weight, int64_t weight_stride0, Rcpp::XPtr<torch::Tensor> weight_buf, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> cx, Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, std::vector<int64_t> batch_sizes, Rcpp::XPtr<torch::Tensor> dropout_state, Rcpp::XPtr<torch::Tensor> reserve, std::vector<bool> output_mask) {
  auto r_out = torch::miopen_rnn_backward(* input, * weight, weight_stride0, * weight_buf, * hx, * cx, * output, * grad_output, * grad_hy, * grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, * dropout_state, * reserve, std_vector_to_std_array<bool,4>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::TensorList>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::mm(* self, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mm_out_out_Tensor_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::mm_out(* out, * self, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_mm_sparse_Tensor_dense_Tensor (Rcpp::XPtr<torch::Tensor> sparse, Rcpp::XPtr<torch::Tensor> dense) {
  auto r_out = torch::_sparse_mm(* sparse, * dense);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::mode(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::mode_out(* values, * indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::mode(* self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_mode_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim) {
  auto r_out = torch::mode_out(* values, * indices, * self, * dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mul_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::mul(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mul_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::mul_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mul_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::mul(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mv_self_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec) {
  auto r_out = torch::mv(* self, * vec);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mv_out_out_Tensor_self_Tensor_vec_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec) {
  auto r_out = torch::mv_out(* out, * self, * vec);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mvlgamma_self_Tensor_p_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t p) {
  auto r_out = torch::mvlgamma(* self, p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_narrow_self_Tensor_dim_int64_t_start_int64_t_length_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t start, int64_t length) {
  auto r_out = torch::narrow(* self, dim, start, length);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_input_Tensor_weight_Tensor_bias_Tensor_running_mean_Tensor_running_var_Tensor_training_bool_momentum_double_eps_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, bool training, double momentum, double eps) {
  auto r_out = torch::native_batch_norm(* input, * weight, * bias, * running_mean, * running_var, training, momentum, eps);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_stats_input_Tensor_eps_double (Rcpp::XPtr<torch::Tensor> input, double eps) {
  auto r_out = torch::batch_norm_stats(* input, eps);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_batch_norm_elemt_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, double eps) {
  auto r_out = torch::batch_norm_elemt(* input, * weight, * bias, * mean, * invstd, eps);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_batch_norm_elemt_out_out_Tensor_input_Tensor_weight_Tensor_bias_Tensor_mean_Tensor_invstd_Tensor_eps_double (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, double eps) {
  auto r_out = torch::batch_norm_elemt_out(* out, * input, * weight, * bias, * mean, * invstd, eps);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_count_int64_t (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, double momentum, double eps, int64_t count) {
  auto r_out = torch::batch_norm_gather_stats(* input, * mean, * invstd, * running_mean, * running_var, momentum, eps, count);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_gather_stats_with_counts_input_Tensor_mean_Tensor_invstd_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double_eps_double_counts_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, double momentum, double eps, std::vector<int64_t> counts) {
  auto r_out = torch::batch_norm_gather_stats_with_counts(* input, * mean, * invstd, * running_mean, * running_var, momentum, eps, counts);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_native_batch_norm_backward_grad_out_Tensor_input_Tensor_weight_Tensor_running_mean_Tensor_running_var_Tensor_save_mean_Tensor_save_invstd_Tensor_train_bool_eps_double_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, Rcpp::XPtr<torch::Tensor> save_mean, Rcpp::XPtr<torch::Tensor> save_invstd, bool train, double eps, std::vector<bool> output_mask) {
  auto r_out = torch::native_batch_norm_backward(* grad_out, * input, * weight, * running_mean, * running_var, * save_mean, * save_invstd, train, eps, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_backward_reduce_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_input_g_bool_weight_g_bool_bias_g_bool (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, Rcpp::XPtr<torch::Tensor> weight, bool input_g, bool weight_g, bool bias_g) {
  auto r_out = torch::batch_norm_backward_reduce(* grad_out, * input, * mean, * invstd, * weight, input_g, weight_g, bias_g);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_batch_norm_backward_elemt_grad_out_Tensor_input_Tensor_mean_Tensor_invstd_Tensor_weight_Tensor_mean_dy_Tensor_mean_dy_xmu_Tensor (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> invstd, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> mean_dy, Rcpp::XPtr<torch::Tensor> mean_dy_xmu) {
  auto r_out = torch::batch_norm_backward_elemt(* grad_out, * input, * mean, * invstd, * weight, * mean_dy, * mean_dy_xmu);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_batch_norm_update_stats_input_Tensor_running_mean_Tensor_running_var_Tensor_momentum_double (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> running_mean, Rcpp::XPtr<torch::Tensor> running_var, double momentum) {
  auto r_out = torch::batch_norm_update_stats(* input, * running_mean, * running_var, momentum);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
bool cpp_torch_namespace__nnpack_available_ () {
  auto r_out = torch::_nnpack_available();
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__nnpack_spatial_convolution_input_Tensor_weight_Tensor_bias_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> weight, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::_nnpack_spatial_convolution(* input, * weight, * bias, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__nnpack_spatial_convolution_backward_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding, std::vector<bool> output_mask) {
  auto r_out = torch::_nnpack_spatial_convolution_backward(* input, * grad_output, * weight, padding, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__nnpack_spatial_convolution_backward_input_input_Tensor_grad_output_Tensor_weight_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> padding) {
  auto r_out = torch::_nnpack_spatial_convolution_backward_input(* input, * grad_output, * weight, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__nnpack_spatial_convolution_backward_weight_input_Tensor_weightsize_IntArrayRef_grad_output_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> input, std::vector<int64_t> weightsize, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> padding) {
  auto r_out = torch::_nnpack_spatial_convolution_backward_weight(* input, weightsize, * grad_output, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ones_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::ones(size, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ones_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::ones(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ones_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size) {
  auto r_out = torch::ones_out(* out, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ones_like_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::ones_like(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ones_like_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::ones_like(* self, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pairwise_distance_x1_Tensor_x2_Tensor (Rcpp::XPtr<torch::Tensor> x1, Rcpp::XPtr<torch::Tensor> x2, double p, double eps, bool keepdim) {
  auto r_out = torch::pairwise_distance(* x1, * x2, p, eps, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cdist_x1_Tensor_x2_Tensor (Rcpp::XPtr<torch::Tensor> x1, Rcpp::XPtr<torch::Tensor> x2, double p, int64_t compute_mode) {
  auto r_out = torch::cdist(* x1, * x2, p, compute_mode);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cdist_backward_grad_Tensor_x1_Tensor_x2_Tensor_p_double_cdist_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> x1, Rcpp::XPtr<torch::Tensor> x2, double p, Rcpp::XPtr<torch::Tensor> cdist) {
  auto r_out = torch::_cdist_backward(* grad, * x1, * x2, p, * cdist);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pdist_self_Tensor (Rcpp::XPtr<torch::Tensor> self, double p) {
  auto r_out = torch::pdist(* self, p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__pdist_forward_self_Tensor (Rcpp::XPtr<torch::Tensor> self, double p) {
  auto r_out = torch::_pdist_forward(* self, p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__pdist_backward_grad_Tensor_self_Tensor_p_double_pdist_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> self, double p, Rcpp::XPtr<torch::Tensor> pdist) {
  auto r_out = torch::_pdist_backward(* grad, * self, p, * pdist);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cosine_similarity_x1_Tensor_x2_Tensor (Rcpp::XPtr<torch::Tensor> x1, Rcpp::XPtr<torch::Tensor> x2, int64_t dim, double eps) {
  auto r_out = torch::cosine_similarity(* x1, * x2, dim, eps);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pixel_shuffle_self_Tensor_upscale_factor_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t upscale_factor) {
  auto r_out = torch::pixel_shuffle(* self, upscale_factor);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pinverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self, double rcond) {
  auto r_out = torch::pinverse(* self, rcond);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_poisson_nll_loss_input_Tensor_target_Tensor_log_input_bool_full_bool_eps_double_reduction_int64_t (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> target, bool log_input, bool full, double eps, int64_t reduction) {
  auto r_out = torch::poisson_nll_loss(* input, * target, log_input, full, eps, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scalar_tensor_s_Scalar (Rcpp::XPtr<torch::Scalar> s, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::scalar_tensor(* s, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::rand(size, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::rand(size, * generator, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::rand(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator (std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::rand(size, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size) {
  auto r_out = torch::rand_out(* out, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_out_out_Tensor_size_IntArrayRef_generator_Generator (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rand_out(* out, size, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_like_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::rand_like(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rand_like_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::rand_like(* self, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef (int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randint(high, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_high_int64_t_size_IntArrayRef_generator_Generator (int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randint(high, size, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef (int64_t low, int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randint(low, high, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (int64_t low, int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randint(low, high, size, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, int64_t high, std::vector<int64_t> size) {
  auto r_out = torch::randint_out(* out, high, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_out_out_Tensor_high_int64_t_size_IntArrayRef_generator_Generator (Rcpp::XPtr<torch::Tensor> out, int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::randint_out(* out, high, size, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, int64_t low, int64_t high, std::vector<int64_t> size) {
  auto r_out = torch::randint_out(* out, low, high, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_out_out_Tensor_low_int64_t_high_int64_t_size_IntArrayRef_generator_Generator (Rcpp::XPtr<torch::Tensor> out, int64_t low, int64_t high, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::randint_out(* out, low, high, size, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_like_self_Tensor_high_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t high, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randint_like(* self, high, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_like_self_Tensor_low_int64_t_high_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t low, int64_t high, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randint_like(* self, low, high, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_like_self_Tensor_high_int64_t_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, int64_t high, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randint_like(* self, high, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randint_like_self_Tensor_low_int64_t_high_int64_t_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, int64_t low, int64_t high, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randint_like(* self, low, high, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randn(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator (std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randn(size, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randn(size, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_size_IntArrayRef_generator_Generator_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randn(size, * generator, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size) {
  auto r_out = torch::randn_out(* out, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_out_out_Tensor_size_IntArrayRef_generator_Generator (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::randn_out(* out, size, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_like_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randn_like(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randn_like_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::randn_like(* self, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randperm_n_int64_t (int64_t n, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randperm(n, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randperm_n_int64_t_generator_Generator (int64_t n, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::randperm(n, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t (Rcpp::XPtr<torch::Tensor> out, int64_t n) {
  auto r_out = torch::randperm_out(* out, n);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_randperm_out_out_Tensor_n_int64_t_generator_Generator (Rcpp::XPtr<torch::Tensor> out, int64_t n, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::randperm_out(* out, n, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_range_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::Scalar> step, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::range(* start, * end, * step, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_range_out_out_Tensor_start_Scalar_end_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> start, Rcpp::XPtr<torch::Scalar> end, Rcpp::XPtr<torch::Scalar> step) {
  auto r_out = torch::range_out(* out, * start, * end, * step);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reciprocal_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::reciprocal(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reciprocal__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::reciprocal_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reciprocal_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::reciprocal_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_neg_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::neg(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_neg__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::neg_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_neg_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::neg_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_repeat_interleave_repeats_Tensor (Rcpp::XPtr<torch::Tensor> repeats) {
  auto r_out = torch::repeat_interleave(* repeats);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> repeats, int64_t dim) {
  auto r_out = torch::repeat_interleave(* self, * repeats, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_repeat_interleave_self_Tensor_repeats_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t repeats, int64_t dim) {
  auto r_out = torch::repeat_interleave(* self, repeats, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> shape) {
  auto r_out = torch::reshape(* self, shape);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__mkldnn_reshape_self_Tensor_shape_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> shape) {
  auto r_out = torch::_mkldnn_reshape(* self, shape);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_round_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::round(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_round__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::round_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_round_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::round_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rrelu(* self, * lower, * upper, training, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rrelu_(* self, * lower, * upper, training, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_relu_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::relu(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_relu__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::relu_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prelu_self_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = torch::prelu(* self, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_prelu_backward_grad_output_Tensor_self_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = torch::prelu_backward(* grad_output, * self, * weight);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gelu_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::gelu(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gelu_backward_grad_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::gelu_backward(* grad, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardshrink_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::hardshrink(* self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardshrink_backward_grad_out_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<torch::Tensor> grad_out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::hardshrink_backward(* grad_out, * self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rsqrt_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::rsqrt(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rsqrt__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::rsqrt_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rsqrt_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::rsqrt_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_select_self_Tensor_dim_Dimname_index_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, int64_t index) {
  auto r_out = torch::select(* self, * dim, index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_select_self_Tensor_dim_int64_t_index_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t index) {
  auto r_out = torch::select(* self, dim, index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_selu_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::selu(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_selu__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::selu_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_celu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::celu(* self, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_celu__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::celu_(* self, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sigmoid_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sigmoid(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sigmoid__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sigmoid_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sigmoid_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sigmoid_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sin_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sin(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sin__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sin_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sin_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sin_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sinh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sinh(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sinh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sinh_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sinh_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sinh_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_detach_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::detach(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_detach__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::detach_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_size_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::size(* self, dim);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_size_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::size(* self, * dim);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slice_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto r_out = torch::slice(* self, dim, start, end, step);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slogdet_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::slogdet(* self);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_smm_self_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::smm(* self, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softmax_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::softmax(* self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softmax_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::softmax(* self, * dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__softmax_self_Tensor_dim_int64_t_half_to_float_bool (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool half_to_float) {
  auto r_out = torch::_softmax(* self, dim, half_to_float);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__softmax_backward_data_grad_output_Tensor_output_Tensor_dim_int64_t_self_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output, int64_t dim, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_softmax_backward_data(* grad_output, * output, dim, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_split_self_Tensor_split_size_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t split_size, int64_t dim) {
  auto r_out = torch::split(* self, split_size, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_split_with_sizes_self_Tensor_split_sizes_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> split_sizes, int64_t dim) {
  auto r_out = torch::split_with_sizes(* self, split_sizes, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_squeeze_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::squeeze(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_squeeze_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::squeeze(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_squeeze_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::squeeze(* self, * dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sspaddmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::sspaddmm(* self, * mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sspaddmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::sspaddmm_out(* out, * self, * mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_stack_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::stack(* tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_stack_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::stack_out(* out, * tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_stft_self_Tensor_n_fft_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t n_fft, int64_t hop_length, int64_t win_length, Rcpp::XPtr<torch::Tensor> window, bool normalized, bool onesided) {
  auto r_out = torch::stft(* self, n_fft, hop_length, win_length, * window, normalized, onesided);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_stride_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::stride(* self, dim);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_stride_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::stride(* self, * dim);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sum_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::sum(* self, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::sum(* self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sum_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::sum(* self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::sum_out(* out, * self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sum_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::sum_out(* out, * self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sqrt_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sqrt(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sqrt__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sqrt_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sqrt_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sqrt_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_std_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::std(* self, unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_std_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std(* self, dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::std_mean(* self, unbiased);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std_mean(* self, dim, unbiased, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_std_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std_mean(* self, * dim, unbiased, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std_out(* out, * self, dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_std_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std(* self, * dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_std_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::std_out(* out, * self, * dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prod_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::prod(* self, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prod_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::prod(* self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::prod_out(* out, * self, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prod_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::prod(* self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_prod_out_out_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::prod_out(* out, * self, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_t_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::t(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tan_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tan(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tan__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tan_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tan_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tan_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tanh_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tanh(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tanh__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tanh_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::tanh_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tensordot_self_Tensor_other_Tensor_dims_self_IntArrayRef_dims_other_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, std::vector<int64_t> dims_self, std::vector<int64_t> dims_other) {
  auto r_out = torch::tensordot(* self, * other, dims_self, dims_other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_threshold_self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> threshold, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::threshold(* self, * threshold, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_threshold__self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> threshold, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::threshold_(* self, * threshold, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_threshold_out_out_Tensor_self_Tensor_threshold_Scalar_value_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> threshold, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::threshold_out(* out, * self, * threshold, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_threshold_backward_grad_output_Tensor_self_Tensor_threshold_Scalar (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> threshold) {
  auto r_out = torch::threshold_backward(* grad_output, * self, * threshold);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim0, int64_t dim1) {
  auto r_out = torch::transpose(* self, dim0, dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_transpose_self_Tensor_dim0_Dimname_dim1_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim0, Rcpp::XPtr<torch::Dimname> dim1) {
  auto r_out = torch::transpose(* self, * dim0, * dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__mkldnn_transpose_self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim0, int64_t dim1) {
  auto r_out = torch::_mkldnn_transpose(* self, dim0, dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__mkldnn_transpose__self_Tensor_dim0_int64_t_dim1_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim0, int64_t dim1) {
  auto r_out = torch::_mkldnn_transpose_(* self, dim0, dim1);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_one_hot_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t num_classes) {
  auto r_out = torch::one_hot(* self, num_classes);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_flip_self_Tensor_dims_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dims) {
  auto r_out = torch::flip(* self, dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_roll_self_Tensor_shifts_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
  auto r_out = torch::roll(* self, shifts, dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rot90_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t k, std::vector<int64_t> dims) {
  auto r_out = torch::rot90(* self, k, dims);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trapz_y_Tensor_x_Tensor (Rcpp::XPtr<torch::Tensor> y, Rcpp::XPtr<torch::Tensor> x, int64_t dim) {
  auto r_out = torch::trapz(* y, * x, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trapz_y_Tensor (Rcpp::XPtr<torch::Tensor> y, double dx, int64_t dim) {
  auto r_out = torch::trapz(* y, dx, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__trilinear_i1_Tensor_i2_Tensor_i3_Tensor_expand1_IntArrayRef_expand2_IntArrayRef_expand3_IntArrayRef_sumdim_IntArrayRef (Rcpp::XPtr<torch::Tensor> i1, Rcpp::XPtr<torch::Tensor> i2, Rcpp::XPtr<torch::Tensor> i3, std::vector<int64_t> expand1, std::vector<int64_t> expand2, std::vector<int64_t> expand3, std::vector<int64_t> sumdim, int64_t unroll_dim) {
  auto r_out = torch::_trilinear(* i1, * i2, * i3, expand1, expand2, expand3, sumdim, unroll_dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_triplet_margin_loss_anchor_Tensor_positive_Tensor_negative_Tensor (Rcpp::XPtr<torch::Tensor> anchor, Rcpp::XPtr<torch::Tensor> positive, Rcpp::XPtr<torch::Tensor> negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  auto r_out = torch::triplet_margin_loss(* anchor, * positive, * negative, margin, p, eps, swap, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trunc_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::trunc(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trunc__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::trunc_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trunc_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::trunc_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace__has_compatible_shallow_copy_type_self_Tensor_from_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> from) {
  auto r_out = torch::_has_compatible_shallow_copy_type(* self, * from);
return r_out;
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool sorted, bool return_inverse) {
  auto r_out = torch::_unique(* self, sorted, return_inverse);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  auto r_out = torch::unique_dim(* self, dim, sorted, return_inverse, return_counts);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_consecutive_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool return_inverse, bool return_counts, int64_t dim) {
  auto r_out = torch::unique_consecutive(* self, return_inverse, return_counts, dim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_unique_dim_consecutive_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool return_inverse, bool return_counts) {
  auto r_out = torch::unique_dim_consecutive(* self, dim, return_inverse, return_counts);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__unique2_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool sorted, bool return_inverse, bool return_counts) {
  auto r_out = torch::_unique2(* self, sorted, return_inverse, return_counts);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__unsafe_view_self_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> size) {
  auto r_out = torch::_unsafe_view(* self, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_unsqueeze_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::unsqueeze(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_var_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::var(* self, unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_var_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var(* self, dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var_out(* out, * self, dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_var_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var(* self, * dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_var_out_out_Tensor_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var_out(* out, * self, * dim, unbiased, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::var_mean(* self, unbiased);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var_mean(* self, dim, unbiased, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_var_mean_self_Tensor_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool unbiased, bool keepdim) {
  auto r_out = torch::var_mean(* self, * dim, unbiased, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> condition, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::where(* condition, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_where_condition_Tensor (Rcpp::XPtr<torch::Tensor> condition) {
  auto r_out = torch::where(* condition);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__s_where_condition_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> condition, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::_s_where(* condition, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_except_dim_v_Tensor (Rcpp::XPtr<torch::Tensor> v, int64_t pow, int64_t dim) {
  auto r_out = torch::norm_except_dim(* v, pow, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__weight_norm_v_Tensor_g_Tensor (Rcpp::XPtr<torch::Tensor> v, Rcpp::XPtr<torch::Tensor> g, int64_t dim) {
  auto r_out = torch::_weight_norm(* v, * g, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_v_Tensor_g_Tensor (Rcpp::XPtr<torch::Tensor> v, Rcpp::XPtr<torch::Tensor> g, int64_t dim) {
  auto r_out = torch::_weight_norm_cuda_interface(* v, * g, dim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_cuda_interface_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> grad_w, Rcpp::XPtr<torch::Tensor> saved_v, Rcpp::XPtr<torch::Tensor> saved_g, Rcpp::XPtr<torch::Tensor> saved_norms, int64_t dim) {
  auto r_out = torch::_weight_norm_cuda_interface_backward(* grad_w, * saved_v, * saved_g, * saved_norms, dim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__weight_norm_differentiable_backward_grad_w_Tensor_saved_v_Tensor_saved_g_Tensor_saved_norms_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> grad_w, Rcpp::XPtr<torch::Tensor> saved_v, Rcpp::XPtr<torch::Tensor> saved_g, Rcpp::XPtr<torch::Tensor> saved_norms, int64_t dim) {
  auto r_out = torch::_weight_norm_differentiable_backward(* grad_w, * saved_v, * saved_g, * saved_norms, dim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zeros_size_IntArrayRef_names_DimnameList (std::vector<int64_t> size, Rcpp::XPtr<std::vector<torch::Dimname>> names, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::zeros(size, * names, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zeros_size_IntArrayRef (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::zeros(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zeros_out_out_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, std::vector<int64_t> size) {
  auto r_out = torch::zeros_out(* out, size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zeros_like_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::zeros_like(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zeros_like_self_Tensor_options_TensorOptions (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::TensorOptions> options, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::zeros_like(* self, * options, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__standard_gamma_grad_self_Tensor_output_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::_standard_gamma_grad(* self, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__standard_gamma_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::_standard_gamma(* self, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__dirichlet_grad_x_Tensor_alpha_Tensor_total_Tensor (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> alpha, Rcpp::XPtr<torch::Tensor> total) {
  auto r_out = torch::_dirichlet_grad(* x, * alpha, * total);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sample_dirichlet_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::_sample_dirichlet(* self, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_poisson_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::poisson(* self, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_native_norm_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p) {
  auto r_out = torch::native_norm(* self, * p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_sum_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_sparse_sum(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_sum_self_Tensor_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::_sparse_sum(* self, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim) {
  auto r_out = torch::_sparse_sum(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_sum_self_Tensor_dim_IntArrayRef_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::_sparse_sum(* self, dim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_sum_backward_grad_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim) {
  auto r_out = torch::_sparse_sum_backward(* grad, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor_p_Scalar_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::norm(* self, * p, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p) {
  auto r_out = torch::norm(* self, * p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::norm(* self, * p, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::norm(* self, * p, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::norm_out(* out, * self, * p, dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::norm_out(* out, * self, * p, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::norm(* self, * p, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::norm(* self, * p, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList_keepdim_bool_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::norm_out(* out, * self, * p, * dim, keepdim, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_norm_out_out_Tensor_self_Tensor_p_Scalar_dim_DimnameList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<std::vector<torch::Dimname>> dim, bool keepdim) {
  auto r_out = torch::norm_out(* out, * self, * p, * dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frobenius_norm_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::frobenius_norm(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frobenius_norm_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::frobenius_norm(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_frobenius_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::frobenius_norm_out(* out, * self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nuclear_norm_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool keepdim) {
  auto r_out = torch::nuclear_norm(* self, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, bool keepdim) {
  auto r_out = torch::nuclear_norm_out(* out, * self, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nuclear_norm_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::nuclear_norm(* self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nuclear_norm_out_out_Tensor_self_Tensor_dim_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> dim, bool keepdim) {
  auto r_out = torch::nuclear_norm_out(* out, * self, dim, keepdim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_clone_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::clone(* self, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_resize_as__self_Tensor_the_template_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> the_template, Rcpp::XPtr<torch::MemoryFormat> memory_format) {
  auto r_out = torch::resize_as_(* self, * the_template, * memory_format);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> exponent) {
  auto r_out = torch::pow_out(* out, * self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_self_Tensor_exponent_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> exponent) {
  auto r_out = torch::pow(* self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_zero__self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::zero_(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sub_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::sub_out(* out, * self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sub_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::sub(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sub_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::sub(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rsub_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::rsub(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rsub_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::rsub(* self, * other, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_addmm_self_Tensor_sparse_Tensor_dense_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> sparse, Rcpp::XPtr<torch::Tensor> dense, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::_sparse_addmm(* self, * sparse, * dense, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addmm_out_out_Tensor_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addmm_out(* out, * self, * mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addmm_self_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addmm(* self, * mat1, * mat2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sparse_coo_tensor_size_IntArrayRef_options_TensorOptions (std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::sparse_coo_tensor(size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor (Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::sparse_coo_tensor(* indices, * values, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sparse_coo_tensor_indices_Tensor_values_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> values, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::sparse_coo_tensor(* indices, * values, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_coo_tensor_unsafe_indices_Tensor_values_Tensor_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> values, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::_sparse_coo_tensor_unsafe(* indices, * values, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_coo_tensor_with_dims_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_options_TensorOptions (int64_t sparse_dim, int64_t dense_dim, std::vector<int64_t> size, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__sparse_coo_tensor_with_dims_and_tensors_sparse_dim_int64_t_dense_dim_int64_t_size_IntArrayRef_indices_Tensor_values_Tensor_options_TensorOptions (int64_t sparse_dim, int64_t dense_dim, std::vector<int64_t> size, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, * indices, * values, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_to_dense_backward_grad_Tensor_input_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> input) {
  auto r_out = torch::to_dense_backward(* grad, * input);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hspmm_out_out_Tensor_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::hspmm_out(* out, * mat1, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hspmm_mat1_Tensor_mat2_Tensor (Rcpp::XPtr<torch::Tensor> mat1, Rcpp::XPtr<torch::Tensor> mat2) {
  auto r_out = torch::hspmm(* mat1, * mat2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_copy_sparse_to_sparse__self_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> src, bool non_blocking) {
  auto r_out = torch::copy_sparse_to_sparse_(* self, * src, non_blocking);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_unbind_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::unbind(* self, dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_unbind_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim) {
  auto r_out = torch::unbind(* self, * dim);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_reorder_conv2d_weight_self_Tensor (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups) {
  auto r_out = torch::mkldnn_reorder_conv2d_weight(* self, padding, stride, dilation, groups);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_to_mkldnn_backward_grad_Tensor_input_Tensor (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> input) {
  auto r_out = torch::to_mkldnn_backward(* grad, * input);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantize_per_tensor_self_Tensor_scale_double_zero_point_int64_t_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, double scale, int64_t zero_point, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::quantize_per_tensor(* self, scale, zero_point, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantize_per_channel_self_Tensor_scales_Tensor_zero_points_Tensor_axis_int64_t_dtype_ScalarType (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> scales, Rcpp::XPtr<torch::Tensor> zero_points, int64_t axis, Rcpp::XPtr<torch::Dtype> dtype) {
  auto r_out = torch::quantize_per_channel(* self, * scales, * zero_points, axis, * dtype);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dequantize_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::dequantize(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
double cpp_torch_namespace_q_scale_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::q_scale(* self);
return r_out;
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_q_zero_point_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::q_zero_point(* self);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_q_per_channel_scales_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::q_per_channel_scales(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_q_per_channel_zero_points_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::q_per_channel_zero_points(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
int64_t cpp_torch_namespace_q_per_channel_axis_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::q_per_channel_axis(* self);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_int_repr_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::int_repr(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__make_per_tensor_quantized_tensor_self_Tensor_scale_double_zero_point_int64_t (Rcpp::XPtr<torch::Tensor> self, double scale, int64_t zero_point) {
  auto r_out = torch::_make_per_tensor_quantized_tensor(* self, scale, zero_point);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__make_per_channel_quantized_tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> scale, Rcpp::XPtr<torch::Tensor> zero_point, int64_t axis) {
  auto r_out = torch::_make_per_channel_quantized_tensor(* self, * scale, * zero_point, axis);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fake_quantize_per_tensor_affine_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<torch::Tensor> self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  auto r_out = torch::fake_quantize_per_tensor_affine(* self, scale, zero_point, quant_min, quant_max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fake_quantize_per_tensor_affine_backward_grad_Tensor_self_Tensor_scale_double_zero_point_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  auto r_out = torch::fake_quantize_per_tensor_affine_backward(* grad, * self, scale, zero_point, quant_min, quant_max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fake_quantize_per_channel_affine_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> scale, Rcpp::XPtr<torch::Tensor> zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  auto r_out = torch::fake_quantize_per_channel_affine(* self, * scale, * zero_point, axis, quant_min, quant_max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fake_quantize_per_channel_affine_backward_grad_Tensor_self_Tensor_scale_Tensor_zero_point_Tensor_axis_int64_t_quant_min_int64_t_quant_max_int64_t (Rcpp::XPtr<torch::Tensor> grad, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> scale, Rcpp::XPtr<torch::Tensor> zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  auto r_out = torch::fake_quantize_per_channel_affine_backward(* grad, * self, * scale, * zero_point, axis, quant_min, quant_max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_meshgrid_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors) {
  auto r_out = torch::meshgrid(* tensors);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cartesian_prod_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors) {
  auto r_out = torch::cartesian_prod(* tensors);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_combinations_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t r, bool with_replacement) {
  auto r_out = torch::combinations(* self, r, with_replacement);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_namespace_result_type_tensor_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> tensor, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::result_type(* tensor, * other);
return make_xptr<torch::Dtype>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_namespace_result_type_tensor_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> tensor, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::result_type(* tensor, * other);
return make_xptr<torch::Dtype>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_namespace_result_type_scalar_Scalar_tensor_Tensor (Rcpp::XPtr<torch::Scalar> scalar, Rcpp::XPtr<torch::Tensor> tensor) {
  auto r_out = torch::result_type(* scalar, * tensor);
return make_xptr<torch::Dtype>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_namespace_result_type_scalar1_Scalar_scalar2_Scalar (Rcpp::XPtr<torch::Scalar> scalar1, Rcpp::XPtr<torch::Scalar> scalar2) {
  auto r_out = torch::result_type(* scalar1, * scalar2);
return make_xptr<torch::Dtype>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_can_cast_from_ScalarType_to_ScalarType (Rcpp::XPtr<torch::Dtype> from, Rcpp::XPtr<torch::Dtype> to) {
  auto r_out = torch::can_cast(* from, * to);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_namespace_promote_types_type1_ScalarType_type2_ScalarType (Rcpp::XPtr<torch::Dtype> type1, Rcpp::XPtr<torch::Dtype> type2) {
  auto r_out = torch::promote_types(* type1, * type2);
return make_xptr<torch::Dtype>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Scalar> cpp_torch_namespace__local_scalar_dense_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_local_scalar_dense(* self);
return make_xptr<torch::Scalar>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_input_gates_Tensor_hidden_gates_Tensor_cx_Tensor (Rcpp::XPtr<torch::Tensor> input_gates, Rcpp::XPtr<torch::Tensor> hidden_gates, Rcpp::XPtr<torch::Tensor> cx, Rcpp::XPtr<torch::Tensor> input_bias, Rcpp::XPtr<torch::Tensor> hidden_bias) {
  auto r_out = torch::_thnn_fused_lstm_cell(* input_gates, * hidden_gates, * cx, * input_bias, * hidden_bias);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_cx_Tensor_cy_Tensor_workspace_Tensor_has_bias_bool (Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> grad_cy, Rcpp::XPtr<torch::Tensor> cx, Rcpp::XPtr<torch::Tensor> cy, Rcpp::XPtr<torch::Tensor> workspace, bool has_bias) {
  auto r_out = torch::_thnn_fused_lstm_cell_backward(* grad_hy, * grad_cy, * cx, * cy, * workspace, has_bias);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_lstm_cell_backward_grad_hy_Tensor_grad_cy_Tensor_input_gates_Tensor_hidden_gates_Tensor_input_bias_Tensor_hidden_bias_Tensor_cx_Tensor_cy_Tensor (Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> grad_cy, Rcpp::XPtr<torch::Tensor> input_gates, Rcpp::XPtr<torch::Tensor> hidden_gates, Rcpp::XPtr<torch::Tensor> input_bias, Rcpp::XPtr<torch::Tensor> hidden_bias, Rcpp::XPtr<torch::Tensor> cx, Rcpp::XPtr<torch::Tensor> cy) {
  auto r_out = torch::_thnn_differentiable_lstm_cell_backward(* grad_hy, * grad_cy, * input_gates, * hidden_gates, * input_bias, * hidden_bias, * cx, * cy);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor (Rcpp::XPtr<torch::Tensor> input_gates, Rcpp::XPtr<torch::Tensor> hidden_gates, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> input_bias, Rcpp::XPtr<torch::Tensor> hidden_bias) {
  auto r_out = torch::_thnn_fused_gru_cell(* input_gates, * hidden_gates, * hx, * input_bias, * hidden_bias);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_fused_gru_cell_backward_grad_hy_Tensor_workspace_Tensor_has_bias_bool (Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> workspace, bool has_bias) {
  auto r_out = torch::_thnn_fused_gru_cell_backward(* grad_hy, * workspace, has_bias);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__thnn_differentiable_gru_cell_backward_grad_hy_Tensor_input_gates_Tensor_hidden_gates_Tensor_hx_Tensor_input_bias_Tensor_hidden_bias_Tensor (Rcpp::XPtr<torch::Tensor> grad_hy, Rcpp::XPtr<torch::Tensor> input_gates, Rcpp::XPtr<torch::Tensor> hidden_gates, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> input_bias, Rcpp::XPtr<torch::Tensor> hidden_bias) {
  auto r_out = torch::_thnn_differentiable_gru_cell_backward(* grad_hy, * input_gates, * hidden_gates, * hx, * input_bias, * hidden_bias);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)),make_xptr<torch::Tensor>(std::get<3>(r_out)),make_xptr<torch::Tensor>(std::get<4>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_input_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = torch::lstm(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_data_Tensor_batch_sizes_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = torch::lstm(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = torch::gru(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_gru_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = torch::gru(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = torch::rnn_tanh(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_tanh_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = torch::rnn_tanh(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = torch::rnn_relu(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_rnn_relu_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = torch::rnn_relu(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh) {
  auto r_out = torch::lstm_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh) {
  auto r_out = torch::gru_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh) {
  auto r_out = torch::rnn_tanh_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh) {
  auto r_out = torch::rnn_relu_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_lstm_input_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, Rcpp::XPtr<torch::Dtype> dtype, bool use_dynamic) {
  auto r_out = torch::quantized_lstm(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first, * dtype, use_dynamic);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_lstm_data_Tensor_batch_sizes_Tensor_hx_TensorList_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, Rcpp::XPtr<torch::Dtype> dtype, bool use_dynamic) {
  auto r_out = torch::quantized_lstm(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, * dtype, use_dynamic);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_gru_input_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto r_out = torch::quantized_gru(* input, * hx, * params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_gru_data_Tensor_batch_sizes_Tensor_hx_Tensor_params_TensorList_has_biases_bool_num_layers_int64_t_dropout_double_train_bool_bidirectional_bool (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<std::vector<torch::Tensor>> params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  auto r_out = torch::quantized_gru(* data, * batch_sizes, * hx, * params, has_biases, num_layers, dropout, train, bidirectional);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_quantized_lstm_cell_input_Tensor_hx_TensorList_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<std::vector<torch::Tensor>> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh, Rcpp::XPtr<torch::Tensor> packed_ih, Rcpp::XPtr<torch::Tensor> packed_hh, Rcpp::XPtr<torch::Tensor> col_offsets_ih, Rcpp::XPtr<torch::Tensor> col_offsets_hh, Rcpp::XPtr<torch::Scalar> scale_ih, Rcpp::XPtr<torch::Scalar> scale_hh, Rcpp::XPtr<torch::Scalar> zero_point_ih, Rcpp::XPtr<torch::Scalar> zero_point_hh) {
  auto r_out = torch::quantized_lstm_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh, * packed_ih, * packed_hh, * col_offsets_ih, * col_offsets_hh, * scale_ih, * scale_hh, * zero_point_ih, * zero_point_hh);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantized_gru_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh, Rcpp::XPtr<torch::Tensor> packed_ih, Rcpp::XPtr<torch::Tensor> packed_hh, Rcpp::XPtr<torch::Tensor> col_offsets_ih, Rcpp::XPtr<torch::Tensor> col_offsets_hh, Rcpp::XPtr<torch::Scalar> scale_ih, Rcpp::XPtr<torch::Scalar> scale_hh, Rcpp::XPtr<torch::Scalar> zero_point_ih, Rcpp::XPtr<torch::Scalar> zero_point_hh) {
  auto r_out = torch::quantized_gru_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh, * packed_ih, * packed_hh, * col_offsets_ih, * col_offsets_hh, * scale_ih, * scale_hh, * zero_point_ih, * zero_point_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantized_rnn_relu_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh, Rcpp::XPtr<torch::Tensor> packed_ih, Rcpp::XPtr<torch::Tensor> packed_hh, Rcpp::XPtr<torch::Tensor> col_offsets_ih, Rcpp::XPtr<torch::Tensor> col_offsets_hh, Rcpp::XPtr<torch::Scalar> scale_ih, Rcpp::XPtr<torch::Scalar> scale_hh, Rcpp::XPtr<torch::Scalar> zero_point_ih, Rcpp::XPtr<torch::Scalar> zero_point_hh) {
  auto r_out = torch::quantized_rnn_relu_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh, * packed_ih, * packed_hh, * col_offsets_ih, * col_offsets_hh, * scale_ih, * scale_hh, * zero_point_ih, * zero_point_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_quantized_rnn_tanh_cell_input_Tensor_hx_Tensor_w_ih_Tensor_w_hh_Tensor_b_ih_Tensor_b_hh_Tensor_packed_ih_Tensor_packed_hh_Tensor_col_offsets_ih_Tensor_col_offsets_hh_Tensor_scale_ih_Scalar_scale_hh_Scalar_zero_point_ih_Scalar_zero_point_hh_Scalar (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> hx, Rcpp::XPtr<torch::Tensor> w_ih, Rcpp::XPtr<torch::Tensor> w_hh, Rcpp::XPtr<torch::Tensor> b_ih, Rcpp::XPtr<torch::Tensor> b_hh, Rcpp::XPtr<torch::Tensor> packed_ih, Rcpp::XPtr<torch::Tensor> packed_hh, Rcpp::XPtr<torch::Tensor> col_offsets_ih, Rcpp::XPtr<torch::Tensor> col_offsets_hh, Rcpp::XPtr<torch::Scalar> scale_ih, Rcpp::XPtr<torch::Scalar> scale_hh, Rcpp::XPtr<torch::Scalar> zero_point_ih, Rcpp::XPtr<torch::Scalar> zero_point_hh) {
  auto r_out = torch::quantized_rnn_tanh_cell(* input, * hx, * w_ih, * w_hh, * b_ih, * b_hh, * packed_ih, * packed_hh, * col_offsets_ih, * col_offsets_hh, * scale_ih, * scale_hh, * zero_point_ih, * zero_point_hh);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pack_padded_sequence_input_Tensor_lengths_Tensor_batch_first_bool (Rcpp::XPtr<torch::Tensor> input, Rcpp::XPtr<torch::Tensor> lengths, bool batch_first) {
  auto r_out = torch::_pack_padded_sequence(* input, * lengths, batch_first);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__pack_padded_sequence_backward_grad_Tensor_input_size_IntArrayRef_batch_sizes_Tensor_batch_first_bool (Rcpp::XPtr<torch::Tensor> grad, std::vector<int64_t> input_size, Rcpp::XPtr<torch::Tensor> batch_sizes, bool batch_first) {
  auto r_out = torch::_pack_padded_sequence_backward(* grad, input_size, * batch_sizes, batch_first);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__pad_packed_sequence_data_Tensor_batch_sizes_Tensor_batch_first_bool_padding_value_Scalar_total_length_int64_t (Rcpp::XPtr<torch::Tensor> data, Rcpp::XPtr<torch::Tensor> batch_sizes, bool batch_first, Rcpp::XPtr<torch::Scalar> padding_value, int64_t total_length) {
  auto r_out = torch::_pad_packed_sequence(* data, * batch_sizes, batch_first, * padding_value, total_length);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::masked_fill(* self, * mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_masked_fill_self_Tensor_mask_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = torch::masked_fill(* self, * mask, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_masked_scatter_self_Tensor_mask_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::masked_scatter(* self, * mask, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_add_self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::index_add(* self, dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_add_self_Tensor_dim_Dimname_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::index_add(* self, * dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::index_fill(* self, dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_fill_self_Tensor_dim_int64_t_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = torch::index_fill(* self, dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::index_fill(* self, * dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_fill_self_Tensor_dim_Dimname_index_Tensor_value_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> value) {
  auto r_out = torch::index_fill(* self, * dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = torch::scatter(* self, dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_self_Tensor_dim_int64_t_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::scatter(* self, dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = torch::scatter(* self, * dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_self_Tensor_dim_Dimname_index_Tensor_value_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::scatter(* self, * dim, * index, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_add_self_Tensor_dim_int64_t_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = torch::scatter_add(* self, dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_scatter_add_self_Tensor_dim_Dimname_index_Tensor_src_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> src) {
  auto r_out = torch::scatter_add(* self, * dim, * index, * src);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___and___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::__and__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___and___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::__and__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___or___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::__or__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___or___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::__or__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::bitwise_xor_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_xor_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::bitwise_xor_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_xor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::bitwise_xor(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_bitwise_xor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::bitwise_xor(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___xor___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::__xor__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___xor___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::__xor__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___lshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::__lshift__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___lshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::__lshift__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___rshift___self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::__rshift__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace___rshift___self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::__rshift__(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addbmm_out_out_Tensor_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addbmm_out(* out, * self, * batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addbmm_self_Tensor_batch1_Tensor_batch2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> batch1, Rcpp::XPtr<torch::Tensor> batch2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::addbmm(* self, * batch1, * batch2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_diag_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::diag_out(* out, * self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_diag_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::diag(* self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cross_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, int64_t dim) {
  auto r_out = torch::cross_out(* out, * self, * other, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cross_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, int64_t dim) {
  auto r_out = torch::cross(* self, * other, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_triu_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::triu_out(* out, * self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_triu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::triu(* self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tril_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::tril_out(* out, * self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tril_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t diagonal) {
  auto r_out = torch::tril(* self, diagonal);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tril_indices_row_int64_t_col_int64_t (int64_t row, int64_t col, int64_t offset, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::tril_indices(row, col, offset, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_triu_indices_row_int64_t_col_int64_t (int64_t row, int64_t col, int64_t offset, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::triu_indices(row, col, offset, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_trace_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::trace(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::ne_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ne_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::ne(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ne_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::ne_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ne_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::ne(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::eq_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eq_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::eq(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eq_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::eq_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_eq_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::eq(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::ge_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ge_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::ge(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ge_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::ge_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ge_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::ge(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::le_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_le_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::le(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_le_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::le_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_le_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::le(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::gt_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gt_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::gt(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gt_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::gt_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gt_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::gt(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::lt_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lt_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::lt(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lt_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::lt_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lt_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::lt(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_take_out_out_Tensor_self_Tensor_index_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::take_out(* out, * self, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_take_self_Tensor_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::take(* self, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::index_select_out(* out, * self, dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_select_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::index_select(* self, dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_select_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::index_select_out(* out, * self, * dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_index_select_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index) {
  auto r_out = torch::index_select(* self, * dim, * index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_masked_select_out_out_Tensor_self_Tensor_mask_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask) {
  auto r_out = torch::masked_select_out(* out, * self, * mask);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_masked_select_self_Tensor_mask_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> mask) {
  auto r_out = torch::masked_select(* self, * mask);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nonzero_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::nonzero_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nonzero_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::nonzero(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorList> cpp_torch_namespace_nonzero_numpy_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::nonzero_numpy(* self);
return make_xptr<torch::TensorList>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = torch::gather_out(* out, * self, dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gather_self_Tensor_dim_int64_t_index_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = torch::gather(* self, dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gather_out_out_Tensor_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = torch::gather_out(* out, * self, * dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_gather_self_Tensor_dim_Dimname_index_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, Rcpp::XPtr<torch::Tensor> index, bool sparse_grad) {
  auto r_out = torch::gather(* self, * dim, * index, sparse_grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__gather_sparse_backward_self_Tensor_dim_int64_t_index_Tensor_grad_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> grad) {
  auto r_out = torch::_gather_sparse_backward(* self, dim, * index, * grad);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addcmul_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::addcmul_out(* out, * self, * tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addcmul_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::addcmul(* self, * tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addcdiv_out_out_Tensor_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::addcdiv_out(* out, * self, * tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_addcdiv_self_Tensor_tensor1_Tensor_tensor2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> tensor1, Rcpp::XPtr<torch::Tensor> tensor2, Rcpp::XPtr<torch::Scalar> value) {
  auto r_out = torch::addcdiv(* self, * tensor1, * tensor2, * value);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_out_X_Tensor_qr_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> X, Rcpp::XPtr<torch::Tensor> qr, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = torch::lstsq_out(* X, * qr, * self, * A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_lstsq_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = torch::lstsq(* self, * A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_out_X_Tensor_M_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> X, Rcpp::XPtr<torch::Tensor> M, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = torch::triangular_solve_out(* X, * M, * self, * A, upper, transpose, unitriangular);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_triangular_solve_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = torch::triangular_solve(* self, * A, upper, transpose, unitriangular);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__triangular_solve_helper_self_Tensor_A_Tensor_upper_bool_transpose_bool_unitriangular_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A, bool upper, bool transpose, bool unitriangular) {
  auto r_out = torch::_triangular_solve_helper(* self, * A, upper, transpose, unitriangular);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_out_e_Tensor_V_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> e, Rcpp::XPtr<torch::Tensor> V, Rcpp::XPtr<torch::Tensor> self, bool eigenvectors, bool upper) {
  auto r_out = torch::symeig_out(* e, * V, * self, eigenvectors, upper);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_symeig_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool eigenvectors, bool upper) {
  auto r_out = torch::symeig(* self, eigenvectors, upper);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__symeig_helper_self_Tensor_eigenvectors_bool_upper_bool (Rcpp::XPtr<torch::Tensor> self, bool eigenvectors, bool upper) {
  auto r_out = torch::_symeig_helper(* self, eigenvectors, upper);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_out_e_Tensor_v_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> e, Rcpp::XPtr<torch::Tensor> v, Rcpp::XPtr<torch::Tensor> self, bool eigenvectors) {
  auto r_out = torch::eig_out(* e, * v, * self, eigenvectors);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_eig_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool eigenvectors) {
  auto r_out = torch::eig(* self, eigenvectors);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_out_U_Tensor_S_Tensor_V_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> U, Rcpp::XPtr<torch::Tensor> S, Rcpp::XPtr<torch::Tensor> V, Rcpp::XPtr<torch::Tensor> self, bool some, bool compute_uv) {
  auto r_out = torch::svd_out(* U, * S, * V, * self, some, compute_uv);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_svd_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool some, bool compute_uv) {
  auto r_out = torch::svd(* self, some, compute_uv);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__svd_helper_self_Tensor_some_bool_compute_uv_bool (Rcpp::XPtr<torch::Tensor> self, bool some, bool compute_uv) {
  auto r_out = torch::_svd_helper(* self, some, compute_uv);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = torch::cholesky_out(* out, * self, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = torch::cholesky(* self, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cholesky_helper_self_Tensor_upper_bool (Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = torch::_cholesky_helper(* self, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_solve_out_out_Tensor_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, bool upper) {
  auto r_out = torch::cholesky_solve_out(* out, * self, * input2, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_solve_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, bool upper) {
  auto r_out = torch::cholesky_solve(* self, * input2, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cholesky_solve_helper_self_Tensor_A_Tensor_upper_bool (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A, bool upper) {
  auto r_out = torch::_cholesky_solve_helper(* self, * A, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = torch::solve(* self, * A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_solve_out_solution_Tensor_lu_Tensor_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> solution, Rcpp::XPtr<torch::Tensor> lu, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = torch::solve_out(* solution, * lu, * self, * A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__solve_helper_self_Tensor_A_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> A) {
  auto r_out = torch::_solve_helper(* self, * A);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_inverse_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = torch::cholesky_inverse_out(* out, * self, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_cholesky_inverse_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool upper) {
  auto r_out = torch::cholesky_inverse(* self, upper);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_out_Q_Tensor_R_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> Q, Rcpp::XPtr<torch::Tensor> R, Rcpp::XPtr<torch::Tensor> self, bool some) {
  auto r_out = torch::qr_out(* Q, * R, * self, some);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_qr_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool some) {
  auto r_out = torch::qr(* self, some);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__qr_helper_self_Tensor_some_bool (Rcpp::XPtr<torch::Tensor> self, bool some) {
  auto r_out = torch::_qr_helper(* self, some);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_out_a_Tensor_tau_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> a, Rcpp::XPtr<torch::Tensor> tau, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::geqrf_out(* a, * tau, * self);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_geqrf_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::geqrf(* self);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_orgqr_out_out_Tensor_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2) {
  auto r_out = torch::orgqr_out(* out, * self, * input2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_orgqr_self_Tensor_input2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2) {
  auto r_out = torch::orgqr(* self, * input2);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ormqr_out_out_Tensor_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> input3, bool left, bool transpose) {
  auto r_out = torch::ormqr_out(* out, * self, * input2, * input3, left, transpose);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_ormqr_self_Tensor_input2_Tensor_input3_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> input2, Rcpp::XPtr<torch::Tensor> input3, bool left, bool transpose) {
  auto r_out = torch::ormqr(* self, * input2, * input3, left, transpose);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__lu_with_info_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool pivot, bool check_errors) {
  auto r_out = torch::_lu_with_info(* self, pivot, check_errors);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lu_solve_out_out_Tensor_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> LU_data, Rcpp::XPtr<torch::Tensor> LU_pivots) {
  auto r_out = torch::lu_solve_out(* out, * self, * LU_data, * LU_pivots);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lu_solve_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> LU_data, Rcpp::XPtr<torch::Tensor> LU_pivots) {
  auto r_out = torch::lu_solve(* self, * LU_data, * LU_pivots);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__lu_solve_helper_self_Tensor_LU_data_Tensor_LU_pivots_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> LU_data, Rcpp::XPtr<torch::Tensor> LU_pivots) {
  auto r_out = torch::_lu_solve_helper(* self, * LU_data, * LU_pivots);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multinomial_out_out_Tensor_self_Tensor_num_samples_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t num_samples, bool replacement, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::multinomial_out(* out, * self, num_samples, replacement, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multinomial_self_Tensor_num_samples_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t num_samples, bool replacement, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::multinomial(* self, num_samples, replacement, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__multinomial_alias_setup_probs_Tensor (Rcpp::XPtr<torch::Tensor> probs) {
  auto r_out = torch::_multinomial_alias_setup(* probs);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__multinomial_alias_draw_J_Tensor_q_Tensor_num_samples_int64_t (Rcpp::XPtr<torch::Tensor> J, Rcpp::XPtr<torch::Tensor> q, int64_t num_samples, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::_multinomial_alias_draw(* J, * q, num_samples, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lgamma_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::lgamma_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lgamma_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::lgamma(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_digamma_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::digamma_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_digamma_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::digamma(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_polygamma_out_out_Tensor_n_int64_t_self_Tensor (Rcpp::XPtr<torch::Tensor> out, int64_t n, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::polygamma_out(* out, n, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_polygamma_n_int64_t_self_Tensor (int64_t n, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::polygamma(n, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erfinv_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erfinv(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_erfinv_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::erfinv_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sign_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sign(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sign_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::sign_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_dist_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other, Rcpp::XPtr<torch::Scalar> p) {
  auto r_out = torch::dist(* self, * other, * p);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_atan2_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::atan2_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_atan2_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::atan2(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Scalar> weight) {
  auto r_out = torch::lerp_out(* out, * self, * end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lerp_out_out_Tensor_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = torch::lerp_out(* out, * self, * end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Scalar> weight) {
  auto r_out = torch::lerp(* self, * end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_lerp_self_Tensor_end_Tensor_weight_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> end, Rcpp::XPtr<torch::Tensor> weight) {
  auto r_out = torch::lerp(* self, * end, * weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_histc_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t bins, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::histc_out(* out, * self, bins, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_histc_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t bins, Rcpp::XPtr<torch::Scalar> min, Rcpp::XPtr<torch::Scalar> max) {
  auto r_out = torch::histc(* self, bins, * min, * max);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::fmod_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fmod_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::fmod(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fmod_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::fmod_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fmod_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::fmod(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::remainder_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_remainder_self_Tensor_other_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> other) {
  auto r_out = torch::remainder(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_remainder_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::remainder_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_remainder_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::remainder(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_min_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::min_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_min_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::min(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_min_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::min(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_out_out_Tensor_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::max_out(* out, * self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::max(* self, * other);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::max(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_median_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::median(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool descending) {
  auto r_out = torch::sort_out(* values, * indices, * self, dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool descending) {
  auto r_out = torch::sort(* self, dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_out_values_Tensor_indices_Tensor_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool descending) {
  auto r_out = torch::sort_out(* values, * indices, * self, * dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_sort_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool descending) {
  auto r_out = torch::sort(* self, * dim, descending);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_argsort_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool descending) {
  auto r_out = torch::argsort(* self, dim, descending);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_argsort_self_Tensor_dim_Dimname (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Dimname> dim, bool descending) {
  auto r_out = torch::argsort(* self, * dim, descending);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_out_values_Tensor_indices_Tensor_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool largest, bool sorted) {
  auto r_out = torch::topk_out(* values, * indices, * self, k, dim, largest, sorted);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_topk_self_Tensor_k_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t k, int64_t dim, bool largest, bool sorted) {
  auto r_out = torch::topk(* self, k, dim, largest, sorted);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_all_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::all(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_any_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::any(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_renorm_out_out_Tensor_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, int64_t dim, Rcpp::XPtr<torch::Scalar> maxnorm) {
  auto r_out = torch::renorm_out(* out, * self, * p, dim, * maxnorm);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_renorm_self_Tensor_p_Scalar_dim_int64_t_maxnorm_Scalar (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> p, int64_t dim, Rcpp::XPtr<torch::Scalar> maxnorm) {
  auto r_out = torch::renorm(* self, * p, dim, * maxnorm);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
bool cpp_torch_namespace_equal_self_Tensor_other_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> other) {
  auto r_out = torch::equal(* self, * other);
return r_out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_out_out_Tensor_self_Tensor_exponent_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = torch::pow_out(* out, * self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_self_Tensor_exponent_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = torch::pow(* self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_out_out_Tensor_self_Scalar_exponent_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Scalar> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = torch::pow_out(* out, * self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_pow_self_Scalar_exponent_Tensor (Rcpp::XPtr<torch::Scalar> self, Rcpp::XPtr<torch::Tensor> exponent) {
  auto r_out = torch::pow(* self, * exponent);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> mean, double std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal_out(* out, * mean, std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_mean_Tensor (Rcpp::XPtr<torch::Tensor> mean, double std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal(* mean, std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_Tensor (Rcpp::XPtr<torch::Tensor> out, double mean, Rcpp::XPtr<torch::Tensor> std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal_out(* out, mean, * std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_mean_double_std_Tensor (double mean, Rcpp::XPtr<torch::Tensor> std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal(mean, * std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_out_out_Tensor_mean_Tensor_std_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal_out(* out, * mean, * std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_mean_Tensor_std_Tensor (Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> std, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal(* mean, * std, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_mean_double_std_double_size_IntArrayRef (double mean, double std, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator, Rcpp::XPtr<torch::TensorOptions> options) {
  auto r_out = torch::normal(mean, std, size, * generator, * options);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_normal_out_out_Tensor_mean_double_std_double_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, double mean, double std, std::vector<int64_t> size, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::normal_out(* out, mean, std, size, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_alias_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::alias(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__addr_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::_addr(* self, * vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__addr__self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::_addr_(* self, * vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__addr_out_out_Tensor_self_Tensor_vec1_Tensor_vec2_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> vec1, Rcpp::XPtr<torch::Tensor> vec2, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> alpha) {
  auto r_out = torch::_addr_out(* out, * self, * vec1, * vec2, * beta, * alpha);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__index_copy__self_Tensor_dim_int64_t_index_Tensor_source_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, Rcpp::XPtr<torch::Tensor> index, Rcpp::XPtr<torch::Tensor> source) {
  auto r_out = torch::_index_copy_(* self, dim, * index, * source);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cumsum_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::_cumsum(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cumsum_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::_cumsum_out(* out, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cumprod_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::_cumprod(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cumprod_out_out_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::_cumprod_out(* out, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__var_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::_var(* self, unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__std_self_Tensor (Rcpp::XPtr<torch::Tensor> self, bool unbiased) {
  auto r_out = torch::_std(* self, unbiased);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cat_tensors_TensorList (Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::_cat(* tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__cat_out_out_Tensor_tensors_TensorList (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<std::vector<torch::Tensor>> tensors, int64_t dim) {
  auto r_out = torch::_cat_out(* out, * tensors, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__mode_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_mode(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__mode_out_values_Tensor_indices_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> values, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_mode_out(* values, * indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__max_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_max(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__max_out_max_Tensor_max_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> max, Rcpp::XPtr<torch::Tensor> max_indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_max_out(* max, * max_indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__min_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_min(* self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace__min_out_min_Tensor_min_indices_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> min, Rcpp::XPtr<torch::Tensor> min_indices, Rcpp::XPtr<torch::Tensor> self, int64_t dim, bool keepdim) {
  auto r_out = torch::_min_out(* min, * min_indices, * self, dim, keepdim);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy_out(* out, * self, * target, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy(* self, * target, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy_backward_out(* grad_input, * grad_output, * self, * target, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_binary_cross_entropy_backward_grad_output_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::binary_cross_entropy_backward(* grad_output, * self, * target, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mse_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::mse_loss_out(* out, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mse_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::mse_loss(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mse_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::mse_loss_backward_out(* grad_input, * grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mse_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::mse_loss_backward(* grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::l1_loss_out(* out, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_l1_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::l1_loss(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::l1_loss_backward_out(* grad_input, * grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::l1_loss_backward(* grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multi_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Scalar> margin, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::multi_margin_loss_out(* out, * self, * target, * p, * margin, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multi_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Scalar> margin, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::multi_margin_loss(* self, * target, * p, * margin, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multi_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Scalar> margin, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::multi_margin_loss_backward_out(* grad_input, * grad_output, * self, * target, * p, * margin, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multi_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_p_Scalar_margin_Scalar (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Scalar> p, Rcpp::XPtr<torch::Scalar> margin, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction) {
  auto r_out = torch::multi_margin_loss_backward(* grad_output, * self, * target, * p, * margin, * weight, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multilabel_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::multilabel_margin_loss_out(* out, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multilabel_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::multilabel_margin_loss(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_out_output_Tensor_is_target_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> is_target, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::multilabel_margin_loss_forward_out(* output, * is_target, * self, * target, reduction);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_multilabel_margin_loss_forward_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::multilabel_margin_loss_forward(* self, * target, reduction);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multilabel_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction, Rcpp::XPtr<torch::Tensor> is_target) {
  auto r_out = torch::multilabel_margin_loss_backward_out(* grad_input, * grad_output, * self, * target, reduction, * is_target);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_multilabel_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t_is_target_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction, Rcpp::XPtr<torch::Tensor> is_target) {
  auto r_out = torch::multilabel_margin_loss_backward(* grad_output, * self, * target, reduction, * is_target);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss_out(* out, * self, * target, * weight, reduction, ignore_index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss(* self, * target, * weight, reduction, ignore_index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> total_weight, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss_forward_out(* output, * total_weight, * self, * target, * weight, reduction, ignore_index);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss_forward(* self, * target, * weight, reduction, ignore_index);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index, Rcpp::XPtr<torch::Tensor> total_weight) {
  auto r_out = torch::nll_loss_backward_out(* grad_input, * grad_output, * self, * target, * weight, reduction, ignore_index, * total_weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index, Rcpp::XPtr<torch::Tensor> total_weight) {
  auto r_out = torch::nll_loss_backward(* grad_output, * self, * target, * weight, reduction, ignore_index, * total_weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss2d_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss2d_out(* out, * self, * target, * weight, reduction, ignore_index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss2d_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss2d(* self, * target, * weight, reduction, ignore_index);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_out_output_Tensor_total_weight_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> total_weight, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss2d_forward_out(* output, * total_weight, * self, * target, * weight, reduction, ignore_index);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_nll_loss2d_forward_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index) {
  auto r_out = torch::nll_loss2d_forward(* self, * target, * weight, reduction, ignore_index);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index, Rcpp::XPtr<torch::Tensor> total_weight) {
  auto r_out = torch::nll_loss2d_backward_out(* grad_input, * grad_output, * self, * target, * weight, reduction, ignore_index, * total_weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_nll_loss2d_backward_grad_output_Tensor_self_Tensor_target_Tensor_weight_Tensor_reduction_int64_t_ignore_index_int64_t_total_weight_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, Rcpp::XPtr<torch::Tensor> weight, int64_t reduction, int64_t ignore_index, Rcpp::XPtr<torch::Tensor> total_weight) {
  auto r_out = torch::nll_loss2d_backward(* grad_output, * self, * target, * weight, reduction, ignore_index, * total_weight);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_smooth_l1_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::smooth_l1_loss_out(* out, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_smooth_l1_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::smooth_l1_loss(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_smooth_l1_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::smooth_l1_loss_backward_out(* grad_input, * grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_smooth_l1_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::smooth_l1_loss_backward(* grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_soft_margin_loss_out_out_Tensor_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::soft_margin_loss_out(* out, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_soft_margin_loss_self_Tensor_target_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::soft_margin_loss(* self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_soft_margin_loss_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::soft_margin_loss_backward_out(* grad_input, * grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_soft_margin_loss_backward_grad_output_Tensor_self_Tensor_target_Tensor_reduction_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> target, int64_t reduction) {
  auto r_out = torch::soft_margin_loss_backward(* grad_output, * self, * target, reduction);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_elu_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> alpha, Rcpp::XPtr<torch::Scalar> scale, Rcpp::XPtr<torch::Scalar> input_scale) {
  auto r_out = torch::elu_out(* out, * self, * alpha, * scale, * input_scale);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_elu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> alpha, Rcpp::XPtr<torch::Scalar> scale, Rcpp::XPtr<torch::Scalar> input_scale) {
  auto r_out = torch::elu(* self, * alpha, * scale, * input_scale);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_elu_backward_out_grad_input_Tensor_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Scalar> alpha, Rcpp::XPtr<torch::Scalar> scale, Rcpp::XPtr<torch::Scalar> input_scale, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::elu_backward_out(* grad_input, * grad_output, * alpha, * scale, * input_scale, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_elu_backward_grad_output_Tensor_alpha_Scalar_scale_Scalar_input_scale_Scalar_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Scalar> alpha, Rcpp::XPtr<torch::Scalar> scale, Rcpp::XPtr<torch::Scalar> input_scale, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::elu_backward(* grad_output, * alpha, * scale, * input_scale, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_elu__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> alpha, Rcpp::XPtr<torch::Scalar> scale, Rcpp::XPtr<torch::Scalar> input_scale) {
  auto r_out = torch::elu_(* self, * alpha, * scale, * input_scale);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_glu_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::glu_out(* out, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_glu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::glu(* self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_glu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::glu_backward_out(* grad_input, * grad_output, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_glu_backward_grad_output_Tensor_self_Tensor_dim_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, int64_t dim) {
  auto r_out = torch::glu_backward(* grad_output, * self, dim);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardtanh_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min_val, Rcpp::XPtr<torch::Scalar> max_val) {
  auto r_out = torch::hardtanh_out(* out, * self, * min_val, * max_val);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardtanh_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min_val, Rcpp::XPtr<torch::Scalar> max_val) {
  auto r_out = torch::hardtanh(* self, * min_val, * max_val);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardtanh_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min_val, Rcpp::XPtr<torch::Scalar> max_val) {
  auto r_out = torch::hardtanh_backward_out(* grad_input, * grad_output, * self, * min_val, * max_val);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardtanh_backward_grad_output_Tensor_self_Tensor_min_val_Scalar_max_val_Scalar (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min_val, Rcpp::XPtr<torch::Scalar> max_val) {
  auto r_out = torch::hardtanh_backward(* grad_output, * self, * min_val, * max_val);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_hardtanh__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> min_val, Rcpp::XPtr<torch::Scalar> max_val) {
  auto r_out = torch::hardtanh_(* self, * min_val, * max_val);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_leaky_relu_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> negative_slope) {
  auto r_out = torch::leaky_relu_out(* out, * self, * negative_slope);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_leaky_relu_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> negative_slope) {
  auto r_out = torch::leaky_relu(* self, * negative_slope);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_leaky_relu_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_negative_slope_Scalar (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> negative_slope) {
  auto r_out = torch::leaky_relu_backward_out(* grad_input, * grad_output, * self, * negative_slope);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_leaky_relu_backward_grad_output_Tensor_self_Tensor_negative_slope_Scalar (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> negative_slope) {
  auto r_out = torch::leaky_relu_backward(* grad_output, * self, * negative_slope);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_leaky_relu__self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> negative_slope) {
  auto r_out = torch::leaky_relu_(* self, * negative_slope);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_sigmoid_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_sigmoid_out(* out, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_sigmoid_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_sigmoid(* self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_out_output_Tensor_buffer_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> buffer, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_sigmoid_forward_out(* output, * buffer, * self);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_log_sigmoid_forward_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::log_sigmoid_forward(* self);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_buffer_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> buffer) {
  auto r_out = torch::log_sigmoid_backward_out(* grad_input, * grad_output, * self, * buffer);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_log_sigmoid_backward_grad_output_Tensor_self_Tensor_buffer_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> buffer) {
  auto r_out = torch::log_sigmoid_backward(* grad_output, * self, * buffer);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_with_noise_out_out_Tensor_self_Tensor_noise_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> noise, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rrelu_with_noise_out(* out, * self, * noise, * lower, * upper, training, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_with_noise_self_Tensor_noise_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> noise, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rrelu_with_noise(* self, * noise, * lower, * upper, training, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_with_noise_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_noise_Tensor_lower_Scalar_upper_Scalar_training_bool (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> noise, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training) {
  auto r_out = torch::rrelu_with_noise_backward_out(* grad_input, * grad_output, * self, * noise, * lower, * upper, training);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_with_noise_backward_grad_output_Tensor_self_Tensor_noise_Tensor_lower_Scalar_upper_Scalar_training_bool (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> noise, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training) {
  auto r_out = torch::rrelu_with_noise_backward(* grad_output, * self, * noise, * lower, * upper, training);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_rrelu_with_noise__self_Tensor_noise_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> noise, Rcpp::XPtr<torch::Scalar> lower, Rcpp::XPtr<torch::Scalar> upper, bool training, Rcpp::XPtr<torch::Generator *> generator) {
  auto r_out = torch::rrelu_with_noise_(* self, * noise, * lower, * upper, training, * generator);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softplus_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> threshold) {
  auto r_out = torch::softplus_out(* out, * self, * beta, * threshold);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softplus_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> threshold) {
  auto r_out = torch::softplus(* self, * beta, * threshold);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softplus_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> threshold, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::softplus_backward_out(* grad_input, * grad_output, * self, * beta, * threshold, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softplus_backward_grad_output_Tensor_self_Tensor_beta_Scalar_threshold_Scalar_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> beta, Rcpp::XPtr<torch::Scalar> threshold, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::softplus_backward(* grad_output, * self, * beta, * threshold, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softshrink_out_out_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::softshrink_out(* out, * self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softshrink_self_Tensor (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::softshrink(* self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softshrink_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::softshrink_backward_out(* grad_input, * grad_output, * self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_softshrink_backward_grad_output_Tensor_self_Tensor_lambd_Scalar (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Scalar> lambd) {
  auto r_out = torch::softshrink_backward(* grad_output, * self, * lambd);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_avg_pool2d_out(* out, * self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_avg_pool2d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_mkldnn_adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::mkldnn_adaptive_avg_pool2d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__adaptive_avg_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::_adaptive_avg_pool2d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__adaptive_avg_pool2d_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::_adaptive_avg_pool2d_backward(* grad_output, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_avg_pool3d_out(* out, * self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_avg_pool3d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::adaptive_avg_pool3d_backward_out(* grad_input, * grad_output, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_avg_pool3d_backward_grad_output_Tensor_self_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::adaptive_avg_pool3d_backward(* grad_output, * self);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_max_pool2d_out(* out, * indices, * self, output_size);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_max_pool2d(* self, output_size);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::adaptive_max_pool2d_backward_out(* grad_input, * grad_output, * self, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_max_pool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::adaptive_max_pool2d_backward(* grad_output, * self, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_out_out_Tensor_indices_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_max_pool3d_out(* out, * indices, * self, output_size);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_adaptive_max_pool3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::adaptive_max_pool3d(* self, output_size);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::adaptive_max_pool3d_backward_out(* grad_input, * grad_output, * self, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_adaptive_max_pool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::adaptive_max_pool3d_backward(* grad_output, * self, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool2d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool2d_out(* out, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool2d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool2d(* self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool2d_backward_out(* grad_input, * grad_output, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool2d_backward(* grad_output, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool3d_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool3d_out(* out, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool3d_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool3d(* self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool3d_backward_out(* grad_input, * grad_output, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_avg_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_ceil_mode_bool_count_include_pad_bool_divisor_override_int64_t (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override) {
  auto r_out = torch::avg_pool3d_backward(* grad_output, * self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> random_samples) {
  auto r_out = torch::fractional_max_pool2d_out(* output, * indices, * self, kernel_size, output_size, * random_samples);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool2d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> random_samples) {
  auto r_out = torch::fractional_max_pool2d(* self, kernel_size, output_size, * random_samples);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fractional_max_pool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::fractional_max_pool2d_backward_out(* grad_input, * grad_output, * self, kernel_size, output_size, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fractional_max_pool2d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::fractional_max_pool2d_backward(* grad_output, * self, kernel_size, output_size, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_out_output_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> random_samples) {
  auto r_out = torch::fractional_max_pool3d_out(* output, * indices, * self, kernel_size, output_size, * random_samples);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_fractional_max_pool3d_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_random_samples_Tensor (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> random_samples) {
  auto r_out = torch::fractional_max_pool3d(* self, kernel_size, output_size, * random_samples);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fractional_max_pool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::fractional_max_pool3d_backward_out(* grad_input, * grad_output, * self, kernel_size, output_size, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_fractional_max_pool3d_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_output_size_IntArrayRef_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> output_size, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::fractional_max_pool3d_backward(* grad_output, * self, kernel_size, output_size, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool2d_with_indices_out(* out, * indices, * self, kernel_size, stride, padding, dilation, ceil_mode);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool2d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool2d_with_indices(* self, kernel_size, stride, padding, dilation, ceil_mode);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool2d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::max_pool2d_with_indices_backward_out(* grad_input, * grad_output, * self, kernel_size, stride, padding, dilation, ceil_mode, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool2d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::max_pool2d_with_indices_backward(* grad_output, * self, kernel_size, stride, padding, dilation, ceil_mode, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_out_out_Tensor_indices_Tensor_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> indices, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool3d_with_indices_out(* out, * indices, * self, kernel_size, stride, padding, dilation, ceil_mode);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_max_pool3d_with_indices_self_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  auto r_out = torch::max_pool3d_with_indices(* self, kernel_size, stride, padding, dilation, ceil_mode);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool3d_with_indices_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::max_pool3d_with_indices_backward_out(* grad_input, * grad_output, * self, kernel_size, stride, padding, dilation, ceil_mode, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_pool3d_with_indices_backward_grad_output_Tensor_self_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_ceil_mode_bool_indices_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Rcpp::XPtr<torch::Tensor> indices) {
  auto r_out = torch::max_pool3d_with_indices_backward(* grad_output, * self, kernel_size, stride, padding, dilation, ceil_mode, * indices);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool2d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size) {
  auto r_out = torch::max_unpool2d_out(* out, * self, * indices, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool2d_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size) {
  auto r_out = torch::max_unpool2d(* self, * indices, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size) {
  auto r_out = torch::max_unpool2d_backward_out(* grad_input, * grad_output, * self, * indices, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool2d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size) {
  auto r_out = torch::max_unpool2d_backward(* grad_output, * self, * indices, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool3d_out_out_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::max_unpool3d_out(* out, * self, * indices, output_size, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool3d_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::max_unpool3d(* self, * indices, output_size, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::max_unpool3d_backward_out(* grad_input, * grad_output, * self, * indices, output_size, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_max_unpool3d_backward_grad_output_Tensor_self_Tensor_indices_Tensor_output_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::max_unpool3d_backward(* grad_output, * self, * indices, output_size, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad1d_out(* out, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad1d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad1d(* self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad1d_backward_out(* grad_input, * grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad1d_backward(* grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad2d_out(* out, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad2d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad2d(* self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad2d_backward_out(* grad_input, * grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_reflection_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::reflection_pad2d_backward(* grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad1d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad1d_out(* out, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad1d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad1d(* self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad1d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad1d_backward_out(* grad_input, * grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad1d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad1d_backward(* grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad2d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad2d_out(* out, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad2d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad2d(* self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad2d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad2d_backward_out(* grad_input, * grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad2d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad2d_backward(* grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad3d_out_out_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad3d_out(* out, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad3d_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad3d(* self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad3d_backward_out_grad_input_Tensor_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad3d_backward_out(* grad_input, * grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_replication_pad3d_backward_grad_output_Tensor_self_Tensor_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> padding) {
  auto r_out = torch::replication_pad3d_backward(* grad_output, * self, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace__test_optional_float_self_Tensor (Rcpp::XPtr<torch::Tensor> self, double scale) {
  auto r_out = torch::_test_optional_float(* self, scale);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_linear1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_linear1d_out(* out, * self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_linear1d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_linear1d(* self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_linear1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_linear1d_backward_out(* grad_input, * grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_linear1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_linear1d_backward(* grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bilinear2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_bilinear2d_out(* out, * self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bilinear2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_bilinear2d(* self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bilinear2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_bilinear2d_backward_out(* grad_input, * grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bilinear2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_bilinear2d_backward(* grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bicubic2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_bicubic2d_out(* out, * self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bicubic2d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_bicubic2d(* self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bicubic2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_bicubic2d_backward_out(* grad_input, * grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_bicubic2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_bicubic2d_backward(* grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_trilinear3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_trilinear3d_out(* out, * self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_trilinear3d_self_Tensor_output_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, bool align_corners) {
  auto r_out = torch::upsample_trilinear3d(* self, output_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_trilinear3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_trilinear3d_backward_out(* grad_input, * grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_trilinear3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef_align_corners_bool (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners) {
  auto r_out = torch::upsample_trilinear3d_backward(* grad_output, output_size, input_size, align_corners);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest1d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest1d_out(* out, * self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest1d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest1d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest1d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest1d_backward_out(* grad_input, * grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest1d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest1d_backward(* grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest2d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest2d_out(* out, * self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest2d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest2d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest2d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest2d_backward_out(* grad_input, * grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest2d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest2d_backward(* grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest3d_out_out_Tensor_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest3d_out(* out, * self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest3d_self_Tensor_output_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size) {
  auto r_out = torch::upsample_nearest3d(* self, output_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest3d_backward_out_grad_input_Tensor_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest3d_backward_out(* grad_input, * grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_upsample_nearest3d_backward_grad_output_Tensor_output_size_IntArrayRef_input_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size) {
  auto r_out = torch::upsample_nearest3d_backward(* grad_output, output_size, input_size);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sigmoid_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::sigmoid_backward_out(* grad_input, * grad_output, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_sigmoid_backward_grad_output_Tensor_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::sigmoid_backward(* grad_output, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tanh_backward_out_grad_input_Tensor_grad_output_Tensor_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::tanh_backward_out(* grad_input, * grad_output, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_tanh_backward_grad_output_Tensor_output_Tensor (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> output) {
  auto r_out = torch::tanh_backward(* grad_output, * output);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_transpose2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_transpose2d_out(* out, * self, * weight, kernel_size, * bias, stride, padding, output_padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_transpose2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_transpose2d(* self, * weight, kernel_size, * bias, stride, padding, output_padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_weight, Rcpp::XPtr<torch::Tensor> grad_bias, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<torch::Tensor> columns, Rcpp::XPtr<torch::Tensor> ones) {
  auto r_out = torch::slow_conv_transpose2d_backward_out(* grad_input, * grad_weight, * grad_bias, * grad_output, * self, * weight, kernel_size, stride, padding, output_padding, dilation, * columns, * ones);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_columns_Tensor_ones_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<torch::Tensor> columns, Rcpp::XPtr<torch::Tensor> ones, std::vector<bool> output_mask) {
  auto r_out = torch::slow_conv_transpose2d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, output_padding, dilation, * columns, * ones, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_transpose3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_transpose3d_out(* out, * self, * weight, kernel_size, * bias, stride, padding, output_padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_transpose3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_transpose3d(* self, * weight, kernel_size, * bias, stride, padding, output_padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_weight, Rcpp::XPtr<torch::Tensor> grad_bias, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input) {
  auto r_out = torch::slow_conv_transpose3d_backward_out(* grad_input, * grad_weight, * grad_bias, * grad_output, * self, * weight, kernel_size, stride, padding, output_padding, dilation, * finput, * fgrad_input);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_transpose3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_output_padding_IntArrayRef_dilation_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = torch::slow_conv_transpose3d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, output_padding, dilation, * finput, * fgrad_input, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::thnn_conv2d_out(* out, * self, * weight, kernel_size, * bias, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::thnn_conv2d(* self, * weight, kernel_size, * bias, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_forward_out_output_Tensor_finput_Tensor_fgrad_input_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::thnn_conv2d_forward_out(* output, * finput, * fgrad_input, * self, * weight, kernel_size, * bias, stride, padding);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::thnn_conv2d_forward(* self, * weight, kernel_size, * bias, stride, padding);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_weight, Rcpp::XPtr<torch::Tensor> grad_bias, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input) {
  auto r_out = torch::thnn_conv2d_backward_out(* grad_input, * grad_weight, * grad_bias, * grad_output, * self, * weight, kernel_size, stride, padding, * finput, * fgrad_input);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = torch::thnn_conv2d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, * finput, * fgrad_input, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv_depthwise2d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::thnn_conv_depthwise2d_out(* out, * self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv_depthwise2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::thnn_conv_depthwise2d(* self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv_depthwise2d_forward_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::thnn_conv_depthwise2d_forward_out(* out, * self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_thnn_conv_depthwise2d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::thnn_conv_depthwise2d_forward(* self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv_depthwise2d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_weight, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::thnn_conv_depthwise2d_backward_out(* grad_input, * grad_weight, * grad_output, * self, * weight, kernel_size, stride, padding, dilation);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_thnn_conv_depthwise2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool2 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = torch::thnn_conv_depthwise2d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, dilation, std_vector_to_std_array<bool,2>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv3d_out_out_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::slow_conv3d_out(* out, * self, * weight, kernel_size, * bias, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::slow_conv3d(* self, * weight, kernel_size, * bias, stride, padding);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_out_output_Tensor_finput_Tensor_fgrad_input_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> output, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::slow_conv3d_forward_out(* output, * finput, * fgrad_input, * self, * weight, kernel_size, * bias, stride, padding);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_forward_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_bias_Tensor_stride_IntArrayRef_padding_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding) {
  auto r_out = torch::slow_conv3d_forward(* self, * weight, kernel_size, * bias, stride, padding);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_out_grad_input_Tensor_grad_weight_Tensor_grad_bias_Tensor_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_weight, Rcpp::XPtr<torch::Tensor> grad_bias, Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input) {
  auto r_out = torch::slow_conv3d_backward_out(* grad_input, * grad_weight, * grad_bias, * grad_output, * self, * weight, kernel_size, stride, padding, * finput, * fgrad_input);
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_finput_Tensor_fgrad_input_Tensor_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, Rcpp::XPtr<torch::Tensor> finput, Rcpp::XPtr<torch::Tensor> fgrad_input, std::vector<bool> output_mask) {
  auto r_out = torch::slow_conv3d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, * finput, * fgrad_input, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_dilated2d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_dilated2d(* self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated2d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = torch::slow_conv_dilated2d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, dilation, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_slow_conv_dilated3d_self_Tensor_weight_Tensor_kernel_size_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, Rcpp::XPtr<torch::Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  auto r_out = torch::slow_conv_dilated3d(* self, * weight, kernel_size, * bias, stride, padding, dilation);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::List cpp_torch_namespace_slow_conv_dilated3d_backward_grad_output_Tensor_self_Tensor_weight_Tensor_kernel_size_IntArrayRef_stride_IntArrayRef_padding_IntArrayRef_dilation_IntArrayRef_output_mask_stdarraybool3 (Rcpp::XPtr<torch::Tensor> grad_output, Rcpp::XPtr<torch::Tensor> self, Rcpp::XPtr<torch::Tensor> weight, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, std::vector<bool> output_mask) {
  auto r_out = torch::slow_conv_dilated3d_backward(* grad_output, * self, * weight, kernel_size, stride, padding, dilation, std_vector_to_std_array<bool,3>(output_mask));
return Rcpp::List::create(make_xptr<torch::Tensor>(std::get<0>(r_out)),make_xptr<torch::Tensor>(std::get<1>(r_out)),make_xptr<torch::Tensor>(std::get<2>(r_out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_col2im_out_out_Tensor_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::col2im_out(* out, * self, output_size, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_col2im_self_Tensor_output_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> output_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::col2im(* self, output_size, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_col2im_backward_out_grad_input_Tensor_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::col2im_backward_out(* grad_input, * grad_output, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_col2im_backward_grad_output_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::col2im_backward(* grad_output, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_im2col_out_out_Tensor_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> out, Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::im2col_out(* out, * self, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_im2col_self_Tensor_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> self, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::im2col(* self, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_im2col_backward_out_grad_input_Tensor_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_input, Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::im2col_backward_out(* grad_input, * grad_output, input_size, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_im2col_backward_grad_output_Tensor_input_size_IntArrayRef_kernel_size_IntArrayRef_dilation_IntArrayRef_padding_IntArrayRef_stride_IntArrayRef (Rcpp::XPtr<torch::Tensor> grad_output, std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> dilation, std::vector<int64_t> padding, std::vector<int64_t> stride) {
  auto r_out = torch::im2col_backward(* grad_output, input_size, kernel_size, dilation, padding, stride);
return make_xptr<torch::Tensor>(r_out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_namespace_isfinite_self_Tensor (Rcpp::XPtr<torch::Tensor> self) {
  auto r_out = torch::isfinite(* self);
return make_xptr<torch::Tensor>(r_out);
}

