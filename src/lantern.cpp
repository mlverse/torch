#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

void lanternTest()
{
    std::cout << "-- Lantern: 0.1.0" << std::endl;

    std::cout << "-- Testing Tensor" << std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "-- Success!" << std::endl;
}

/* Autogen Body -- Start */
void lantern__cast_byte_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_char_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_double_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_float_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_int_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_long_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_short_tensor_bool(void* self, void* non_blocking) {}
void lantern__cast_half_tensor_bool(void* self, void* non_blocking) {}
void lantern_backward_tensor_tensor_bool_bool(void* self, void* gradient, void* keep_graph, void* create_graph) {}
void lantern_set_data_tensor_tensor(void* self, void* new_data) {}
void lantern_data_tensor(void* self) {}
void lantern_is_leaf_tensor(void* self) {}
void lantern_output_nr_tensor(void* self) {}
void lantern__version_tensor(void* self) {}
void lantern_rename__tensor_dimnamelist(void* self, void* names) {}
void lantern_rename_tensor_dimnamelist(void* self, void* names) {}
void lantern_align_to_tensor_dimnamelist(void* self, void* names) {}
void lantern_align_as_tensor_tensor(void* self, void* other) {}
void lantern_align_tensors_tensorlist(void* tensors) {}
void lantern_refine_names_tensor_dimnamelist(void* self, void* names) {}
void lantern_unflatten_tensor_dimname_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names) {}
void lantern_unflatten_tensor_intt_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names) {}
void lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* deterministic, void* zero_infinity) {}
void lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_bool_bool(void* weight_arr, void* weight_stride0, void* input_size, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* bidirectional) {}
void lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state) {}
void lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask) {}
void lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions(void* dropout, void* train, void* dropout_seed, void* options) {}
void lantern__debug_has_internal_overlap_tensor(void* self) {}
void lantern__fused_dropout_tensor_double_generator(void* self, void* p, void* generator) {}
void lantern__masked_scale_tensor_tensor_double(void* self, void* mask, void* scale) {}
void lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype(void* quasi, void* n, void* sobolstate, void* dimension, void* num_generated, void* dtype) {}
void lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(void* self, void* n, void* sobolstate, void* dimension, void* num_generated) {}
void lantern__sobol_engine_scramble__tensor_tensor_intt(void* self, void* ltm, void* dimension) {}
void lantern__sobol_engine_initialize_state__tensor_intt(void* self, void* dimension) {}
void lantern__reshape_from_tensor_tensor_tensor(void* self, void* shape) {}
void lantern__shape_as_tensor_tensor(void* self) {}
void lantern_dropout_tensor_double_bool(void* input, void* p, void* train) {}
void lantern_dropout__tensor_double_bool(void* self, void* p, void* train) {}
void lantern_feature_dropout_tensor_double_bool(void* input, void* p, void* train) {}
void lantern_feature_dropout__tensor_double_bool(void* self, void* p, void* train) {}
void lantern_alpha_dropout_tensor_double_bool(void* input, void* p, void* train) {}
void lantern_alpha_dropout__tensor_double_bool(void* self, void* p, void* train) {}
void lantern_feature_alpha_dropout_tensor_double_bool(void* input, void* p, void* train) {}
void lantern_feature_alpha_dropout__tensor_double_bool(void* self, void* p, void* train) {}
void lantern_abs_tensor(void* self) {}
void lantern_abs__tensor(void* self) {}
void lantern_abs_out_tensor_tensor(void* out, void* self) {}
void lantern_acos_tensor(void* self) {}
void lantern_acos__tensor(void* self) {}
void lantern_acos_out_tensor_tensor(void* out, void* self) {}
void lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad) {}
void lantern_adaptive_avg_pool1d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_adaptive_max_pool1d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_add_tensor_tensor_scalar(void* self, void* other, void* alpha) {}
void lantern_add__tensor_tensor_scalar(void* self, void* other, void* alpha) {}
void lantern_add_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha) {}
void lantern_add_tensor_scalar_scalar(void* self, void* other, void* alpha) {}
void lantern_add__tensor_scalar_scalar(void* self, void* other, void* alpha) {}
void lantern_addmv_tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha) {}
void lantern_addmv__tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha) {}
void lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat, void* vec, void* beta, void* alpha) {}
void lantern_addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern_addr__tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern_affine_grid_generator_tensor_intarrayref_bool(void* theta, void* size, void* align_corners) {}
void lantern_affine_grid_generator_backward_tensor_intarrayref_bool(void* grad, void* size, void* align_corners) {}
void lantern_all_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_all_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_all_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_all_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_allclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan) {}
void lantern_any_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_any_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_any_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_any_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_arange_scalar_tensoroptions(void* end, void* options) {}
void lantern_arange_scalar_scalar_tensoroptions(void* start, void* end, void* options) {}
void lantern_arange_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options) {}
void lantern_arange_out_tensor_scalar(void* out, void* end) {}
void lantern_arange_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step) {}
void lantern__dim_arange_tensor_intt(void* like, void* dim) {}
void lantern_argmax_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_argmin_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_as_strided_tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset) {}
void lantern_as_strided__tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset) {}
void lantern_asin_tensor(void* self) {}
void lantern_asin__tensor(void* self) {}
void lantern_asin_out_tensor_tensor(void* out, void* self) {}
void lantern_atan_tensor(void* self) {}
void lantern_atan__tensor(void* self) {}
void lantern_atan_out_tensor_tensor(void* out, void* self) {}
void lantern_baddbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_baddbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_bartlett_window_intt_tensoroptions(void* window_length, void* options) {}
void lantern_bartlett_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options) {}
void lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled) {}
void lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled) {}
void lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(void* impl_index, void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var_transform, void* train, void* eps, void* output_mask) {}
void lantern_bernoulli_tensor_generator(void* self, void* generator) {}
void lantern_bernoulli_out_tensor_tensor_generator(void* out, void* self, void* generator) {}
void lantern_bernoulli__tensor_tensor_generator(void* self, void* p, void* generator) {}
void lantern_bernoulli__tensor_double_generator(void* self, void* p, void* generator) {}
void lantern_bernoulli_tensor_double_generator(void* self, void* p, void* generator) {}
void lantern_bilinear_tensor_tensor_tensor_tensor(void* input1, void* input2, void* weight, void* bias) {}
void lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* pos_weight, void* reduction) {}
void lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction) {}
void lantern_bincount_tensor_tensor_intt(void* self, void* weights, void* minlength) {}
void lantern_bitwise_not_tensor(void* self) {}
void lantern_bitwise_not__tensor(void* self) {}
void lantern_bitwise_not_out_tensor_tensor(void* out, void* self) {}
void lantern_logical_not_tensor(void* self) {}
void lantern_logical_not__tensor(void* self) {}
void lantern_logical_not_out_tensor_tensor(void* out, void* self) {}
void lantern_logical_xor_tensor_tensor(void* self, void* other) {}
void lantern_logical_xor__tensor_tensor(void* self, void* other) {}
void lantern_logical_xor_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_blackman_window_intt_tensoroptions(void* window_length, void* options) {}
void lantern_blackman_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options) {}
void lantern_bmm_tensor_tensor(void* self, void* mat2) {}
void lantern_bmm_out_tensor_tensor_tensor(void* out, void* self, void* mat2) {}
void lantern_broadcast_tensors_tensorlist(void* tensors) {}
void lantern_cat_tensorlist_intt(void* tensors, void* dim) {}
void lantern_cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim) {}
void lantern_cat_tensorlist_dimname(void* tensors, void* dim) {}
void lantern_cat_out_tensor_tensorlist_dimname(void* out, void* tensors, void* dim) {}
void lantern_ceil_tensor(void* self) {}
void lantern_ceil__tensor(void* self) {}
void lantern_ceil_out_tensor_tensor(void* out, void* self) {}
void lantern_chain_matmul_tensorlist(void* matrices) {}
void lantern_chunk_tensor_intt_intt(void* self, void* chunks, void* dim) {}
void lantern_clamp_tensor_scalar_scalar(void* self, void* min, void* max) {}
void lantern_clamp__tensor_scalar_scalar(void* self, void* min, void* max) {}
void lantern_clamp_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min, void* max) {}
void lantern_clamp_max_tensor_scalar(void* self, void* max) {}
void lantern_clamp_max__tensor_scalar(void* self, void* max) {}
void lantern_clamp_max_out_tensor_tensor_scalar(void* out, void* self, void* max) {}
void lantern_clamp_min_tensor_scalar(void* self, void* min) {}
void lantern_clamp_min__tensor_scalar(void* self, void* min) {}
void lantern_clamp_min_out_tensor_tensor_scalar(void* out, void* self, void* min) {}
void lantern_cudnn_is_acceptable_tensor(void* self) {}
void lantern_constant_pad_nd_tensor_intarrayref_scalar(void* self, void* pad, void* value) {}
void lantern_contiguous_tensor_memoryformat(void* self, void* memory_format) {}
void lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups) {}
void lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups) {}
void lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool(void* grad_output, void* input, void* weight, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* output_mask) {}
void lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled) {}
void lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding) {}
void lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_stdarraybool(void* ggI, void* ggW, void* ggb, void* gO, void* weight, void* self, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* output_mask) {}
void lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups) {}
void lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups) {}
void lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups) {}
void lantern_conv_tbc_tensor_tensor_tensor_intt(void* self, void* weight, void* bias, void* pad) {}
void lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt(void* self, void* input, void* weight, void* bias, void* pad) {}
void lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation) {}
void lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation) {}
void lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation) {}
void lantern_copy__tensor_tensor_bool(void* self, void* src, void* non_blocking) {}
void lantern__copy_from_tensor_tensor_bool(void* self, void* dst, void* non_blocking) {}
void lantern_cos_tensor(void* self) {}
void lantern_cos__tensor(void* self) {}
void lantern_cos_out_tensor_tensor(void* out, void* self) {}
void lantern_cosh_tensor(void* self) {}
void lantern_cosh__tensor(void* self) {}
void lantern_cosh_out_tensor_tensor(void* out, void* self) {}
void lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction) {}
void lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt(void* theta, void* N, void* C, void* H, void* W) {}
void lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt(void* grad, void* N, void* C, void* H, void* W) {}
void lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon) {}
void lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon) {}
void lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask) {}
void lantern_cudnn_convolution_backward_bias_tensor(void* grad_output) {}
void lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask) {}
void lantern_cudnn_convolution_transpose_backward_bias_tensor(void* grad_output) {}
void lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_cudnn_grid_sampler_tensor_tensor(void* self, void* grid) {}
void lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor(void* self, void* grid, void* grad_output) {}
void lantern_cumsum_tensor_intt_scalartype(void* self, void* dim, void* dtype) {}
void lantern_cumsum_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype) {}
void lantern_cumsum_tensor_dimname_scalartype(void* self, void* dim, void* dtype) {}
void lantern_cumsum_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype) {}
void lantern_cumprod_tensor_intt_scalartype(void* self, void* dim, void* dtype) {}
void lantern_cumprod_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype) {}
void lantern_cumprod_tensor_dimname_scalartype(void* self, void* dim, void* dtype) {}
void lantern_cumprod_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype) {}
void lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity) {}
void lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity) {}
void lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* zero_infinity) {}
void lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool(void* grad, void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* neg_log_likelihood, void* log_alpha, void* blank, void* zero_infinity) {}
void lantern_det_tensor(void* self) {}
void lantern_diag_embed_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2) {}
void lantern_diagflat_tensor_intt(void* self, void* offset) {}
void lantern_diagonal_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2) {}
void lantern_fill_diagonal__tensor_scalar_bool(void* self, void* fill_value, void* wrap) {}
void lantern_div_tensor_tensor(void* self, void* other) {}
void lantern_div__tensor_tensor(void* self, void* other) {}
void lantern_div_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_div_tensor_scalar(void* self, void* other) {}
void lantern_div__tensor_scalar(void* self, void* other) {}
void lantern_dot_tensor_tensor(void* self, void* tensor) {}
void lantern_dot_out_tensor_tensor_tensor(void* out, void* self, void* tensor) {}
void lantern_einsum_stdstring_tensorlist(void* equation, void* tensors) {}
void lantern_embedding_tensor_tensor_intt_bool_bool(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse) {}
void lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse) {}
void lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool(void* grad_output, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq) {}
void lantern_embedding_renorm__tensor_tensor_double_double(void* self, void* indices, void* max_norm, void* norm_type) {}
void lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq) {}
void lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights) {}
void lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights) {}
void lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights) {}
void lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights) {}
void lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights) {}
void lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad, void* weight, void* indices, void* offsets, void* offset2bag, void* mode) {}
void lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(void* size, void* names, void* options, void* memory_format) {}
void lantern_empty_intarrayref_tensoroptions_memoryformat(void* size, void* options, void* memory_format) {}
void lantern_new_empty_tensor_intarrayref_tensoroptions(void* self, void* size, void* options) {}
void lantern_new_full_tensor_intarrayref_scalar_tensoroptions(void* self, void* size, void* fill_value, void* options) {}
void lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat(void* size, void* options, void* scale, void* zero_point, void* memory_format) {}
void lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat(void* size, void* scales, void* zero_points, void* axis, void* options, void* memory_format) {}
void lantern_resize__tensor_intarrayref(void* self, void* size) {}
void lantern_empty_out_tensor_intarrayref_memoryformat(void* out, void* size, void* memory_format) {}
void lantern_empty_like_tensor(void* self) {}
void lantern_empty_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format) {}
void lantern_empty_strided_intarrayref_intarrayref_tensoroptions(void* size, void* stride, void* options) {}
void lantern_erf_tensor(void* self) {}
void lantern_erf__tensor(void* self) {}
void lantern_erf_out_tensor_tensor(void* out, void* self) {}
void lantern_erfc_tensor(void* self) {}
void lantern_erfc__tensor(void* self) {}
void lantern_erfc_out_tensor_tensor(void* out, void* self) {}
void lantern_exp_tensor(void* self) {}
void lantern_exp__tensor(void* self) {}
void lantern_exp_out_tensor_tensor(void* out, void* self) {}
void lantern_expm1_tensor(void* self) {}
void lantern_expm1__tensor(void* self) {}
void lantern_expm1_out_tensor_tensor(void* out, void* self) {}
void lantern_expand_tensor_intarrayref_bool(void* self, void* size, void* implicit) {}
void lantern_expand_as_tensor_tensor(void* self, void* other) {}
void lantern_eye_intt_tensoroptions(void* n, void* options) {}
void lantern_eye_intt_intt_tensoroptions(void* n, void* m, void* options) {}
void lantern_eye_out_tensor_intt(void* out, void* n) {}
void lantern_eye_out_tensor_intt_intt(void* out, void* n, void* m) {}
void lantern_flatten_tensor_intt_intt(void* self, void* start_dim, void* end_dim) {}
void lantern_flatten_tensor_intt_intt_dimname(void* self, void* start_dim, void* end_dim, void* out_dim) {}
void lantern_flatten_tensor_dimname_dimname_dimname(void* self, void* start_dim, void* end_dim, void* out_dim) {}
void lantern_flatten_tensor_dimnamelist_dimname(void* self, void* dims, void* out_dim) {}
void lantern_fill__tensor_scalar(void* self, void* value) {}
void lantern_fill__tensor_tensor(void* self, void* value) {}
void lantern_floor_tensor(void* self) {}
void lantern_floor__tensor(void* self) {}
void lantern_floor_out_tensor_tensor(void* out, void* self) {}
void lantern_frac_tensor(void* self) {}
void lantern_frac__tensor(void* self) {}
void lantern_frac_out_tensor_tensor(void* out, void* self) {}
void lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(void* size, void* fill_value, void* names, void* options) {}
void lantern_full_intarrayref_scalar_tensoroptions(void* size, void* fill_value, void* options) {}
void lantern_full_out_tensor_intarrayref_scalar(void* out, void* size, void* fill_value) {}
void lantern_full_like_tensor_scalar(void* self, void* fill_value) {}
void lantern_full_like_tensor_scalar_tensoroptions(void* self, void* fill_value, void* options) {}
void lantern_from_file_stdstring_bool_intt_tensoroptions(void* filename, void* shared, void* size, void* options) {}
void lantern_grid_sampler_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners) {}
void lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners) {}
void lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners) {}
void lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners) {}
void lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners) {}
void lantern_hann_window_intt_tensoroptions(void* window_length, void* options) {}
void lantern_hann_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options) {}
void lantern_hamming_window_intt_tensoroptions(void* window_length, void* options) {}
void lantern_hamming_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options) {}
void lantern_hamming_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* options) {}
void lantern_hamming_window_intt_bool_double_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* beta, void* options) {}
void lantern_hinge_embedding_loss_tensor_tensor_double_intt(void* self, void* target, void* margin, void* reduction) {}
void lantern_ger_tensor_tensor(void* self, void* vec2) {}
void lantern_ger_out_tensor_tensor_tensor(void* out, void* self, void* vec2) {}
void lantern_group_norm_tensor_intt_tensor_tensor_double_bool(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled) {}
void lantern_fft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized) {}
void lantern_ifft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized) {}
void lantern_rfft_tensor_intt_bool_bool(void* self, void* signal_ndim, void* normalized, void* onesided) {}
void lantern_irfft_tensor_intt_bool_bool_intarrayref(void* self, void* signal_ndim, void* normalized, void* onesided, void* signal_sizes) {}
void lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_bool_bool_intarrayref(void* self, void* signal_ndim, void* complex_input, void* complex_output, void* inverse, void* checked_signal_sizes, void* normalized, void* onesided, void* output_sizes) {}
void lantern__cufft_get_plan_cache_size_intt(void* device_index) {}
void lantern__cufft_get_plan_cache_max_size_intt(void* device_index) {}
void lantern__cufft_set_plan_cache_max_size_intt_intt(void* device_index, void* max_size) {}
void lantern__cufft_clear_plan_cache_intt(void* device_index) {}
void lantern_index_tensor_tensorlist(void* self, void* indices) {}
void lantern_index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_copy_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_copy__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_copy_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_put__tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate) {}
void lantern_index_put_tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate) {}
void lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool(void* self, void* indices, void* values, void* accumulate, void* unsafe) {}
void lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled) {}
void lantern_inverse_tensor(void* self) {}
void lantern_inverse_out_tensor_tensor(void* out, void* self) {}
void lantern__inverse_helper_tensor(void* self) {}
void lantern_isclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan) {}
void lantern_isnan_tensor(void* self) {}
void lantern_is_distributed_tensor(void* self) {}
void lantern_is_floating_point_tensor(void* self) {}
void lantern_is_complex_tensor(void* self) {}
void lantern_is_nonzero_tensor(void* self) {}
void lantern_is_same_size_tensor_tensor(void* self, void* other) {}
void lantern_is_signed_tensor(void* self) {}
void lantern_kl_div_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_kl_div_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction) {}
void lantern_kthvalue_tensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim) {}
void lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim) {}
void lantern_kthvalue_tensor_intt_dimname_bool(void* self, void* k, void* dim, void* keepdim) {}
void lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim) {}
void lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable) {}
void lantern_native_layer_norm_tensor_tensor_tensor_intt_intt_double(void* input, void* weight, void* bias, void* M, void* N, void* eps) {}
void lantern_native_layer_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* M, void* N, void* output_mask) {}
void lantern_native_layer_norm_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool(void* ggI, void* ggW, void* ggb, void* gO, void* input, void* mean, void* rstd, void* weight, void* M, void* N, void* output_mask) {}
void lantern_linear_tensor_tensor_tensor(void* input, void* weight, void* bias) {}
void lantern_mkldnn_linear_tensor_tensor_tensor(void* input, void* weight, void* bias) {}
void lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias) {}
void lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias) {}
void lantern_fbgemm_linear_quantize_weight_tensor(void* input) {}
void lantern_fbgemm_pack_gemm_matrix_fp16_tensor(void* input) {}
void lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(void* input, void* packed_weight, void* bias) {}
void lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(void* input, void* packed_weight, void* bias) {}
void lantern_fbgemm_pack_quantized_matrix_tensor(void* input) {}
void lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(void* input, void* K, void* N) {}
void lantern_linspace_scalar_scalar_intt_tensoroptions(void* start, void* end, void* steps, void* options) {}
void lantern_linspace_out_tensor_scalar_scalar_intt(void* out, void* start, void* end, void* steps) {}
void lantern_log_tensor(void* self) {}
void lantern_log__tensor(void* self) {}
void lantern_log_out_tensor_tensor(void* out, void* self) {}
void lantern_log10_tensor(void* self) {}
void lantern_log10__tensor(void* self) {}
void lantern_log10_out_tensor_tensor(void* out, void* self) {}
void lantern_log1p_tensor(void* self) {}
void lantern_log1p__tensor(void* self) {}
void lantern_log1p_out_tensor_tensor(void* out, void* self) {}
void lantern_log2_tensor(void* self) {}
void lantern_log2__tensor(void* self) {}
void lantern_log2_out_tensor_tensor(void* out, void* self) {}
void lantern_logdet_tensor(void* self) {}
void lantern_logspace_scalar_scalar_intt_double_tensoroptions(void* start, void* end, void* steps, void* base, void* options) {}
void lantern_logspace_out_tensor_scalar_scalar_intt_double(void* out, void* start, void* end, void* steps, void* base) {}
void lantern_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype) {}
void lantern_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype) {}
void lantern__log_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float) {}
void lantern__log_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self) {}
void lantern_logsumexp_tensor_intarrayref_bool(void* self, void* dim, void* keepdim) {}
void lantern_logsumexp_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_logsumexp_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim) {}
void lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction) {}
void lantern_matmul_tensor_tensor(void* self, void* other) {}
void lantern_matmul_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_matrix_rank_tensor_double_bool(void* self, void* tol, void* symmetric) {}
void lantern_matrix_rank_tensor_bool(void* self, void* symmetric) {}
void lantern_matrix_power_tensor_intt(void* self, void* n) {}
void lantern_max_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_max_out_tensor_tensor_tensor_intt_bool(void* max, void* max_values, void* self, void* dim, void* keepdim) {}
void lantern_max_values_tensor_intarrayref_bool(void* self, void* dim, void* keepdim) {}
void lantern_max_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_max_out_tensor_tensor_tensor_dimname_bool(void* max, void* max_values, void* self, void* dim, void* keepdim) {}
void lantern_max_values_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim) {}
void lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_mean_tensor_scalartype(void* self, void* dtype) {}
void lantern_mean_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_mean_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_median_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_median_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim) {}
void lantern_median_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_median_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim) {}
void lantern_min_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_min_out_tensor_tensor_tensor_intt_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim) {}
void lantern_min_values_tensor_intarrayref_bool(void* self, void* dim, void* keepdim) {}
void lantern_min_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_min_out_tensor_tensor_tensor_dimname_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim) {}
void lantern_min_values_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim) {}
void lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups) {}
void lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined) {}
void lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* bias_defined) {}
void lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* output_mask) {}
void lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon) {}
void lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon) {}
void lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask) {}
void lantern_miopen_convolution_backward_bias_tensor(void* grad_output) {}
void lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask) {}
void lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask) {}
void lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic) {}
void lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state) {}
void lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask) {}
void lantern_mm_tensor_tensor(void* self, void* mat2) {}
void lantern_mm_out_tensor_tensor_tensor(void* out, void* self, void* mat2) {}
void lantern__sparse_mm_tensor_tensor(void* sparse, void* dense) {}
void lantern_mode_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern_mode_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim) {}
void lantern_mode_tensor_dimname_bool(void* self, void* dim, void* keepdim) {}
void lantern_mode_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim) {}
void lantern_mul_tensor_tensor(void* self, void* other) {}
void lantern_mul__tensor_tensor(void* self, void* other) {}
void lantern_mul_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_mul_tensor_scalar(void* self, void* other) {}
void lantern_mul__tensor_scalar(void* self, void* other) {}
void lantern_mv_tensor_tensor(void* self, void* vec) {}
void lantern_mv_out_tensor_tensor_tensor(void* out, void* self, void* vec) {}
void lantern_mvlgamma_tensor_intt(void* self, void* p) {}
void lantern_mvlgamma__tensor_intt(void* self, void* p) {}
void lantern_narrow_copy_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length) {}
void lantern_narrow_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length) {}
void lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps) {}
void lantern_batch_norm_stats_tensor_double(void* input, void* eps) {}
void lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double(void* input, void* weight, void* bias, void* mean, void* invstd, void* eps) {}
void lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* count) {}
void lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_intarrayref(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* counts) {}
void lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(void* grad_out, void* input, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_invstd, void* train, void* eps, void* output_mask) {}
void lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* input_g, void* weight_g, void* bias_g) {}
void lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* mean_dy, void* mean_dy_xmu) {}
void lantern_batch_norm_update_stats_tensor_tensor_tensor_double(void* input, void* running_mean, void* running_var, void* momentum) {}
void lantern__nnpack_available() {}
void lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref(void* input, void* weight, void* bias, void* padding) {}
void lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool(void* input, void* grad_output, void* weight, void* padding, void* output_mask) {}
void lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(void* input, void* grad_output, void* weight, void* padding) {}
void lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(void* input, void* weightsize, void* grad_output, void* padding) {}
void lantern_ones_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options) {}
void lantern_ones_intarrayref_tensoroptions(void* size, void* options) {}
void lantern_ones_out_tensor_intarrayref(void* out, void* size) {}
void lantern_ones_like_tensor(void* self) {}
void lantern_ones_like_tensor_tensoroptions(void* self, void* options) {}
void lantern_pairwise_distance_tensor_tensor_double_double_bool(void* x1, void* x2, void* p, void* eps, void* keepdim) {}
void lantern_cdist_tensor_tensor_double(void* x1, void* x2, void* p) {}
void lantern__cdist_backward_tensor_tensor_tensor_double_tensor(void* grad, void* x1, void* x2, void* p, void* cdist) {}
void lantern_pdist_tensor_double(void* self, void* p) {}
void lantern__pdist_forward_tensor_double(void* self, void* p) {}
void lantern__pdist_backward_tensor_tensor_double_tensor(void* grad, void* self, void* p, void* pdist) {}
void lantern_cosine_similarity_tensor_tensor_intt_double(void* x1, void* x2, void* dim, void* eps) {}
void lantern_permute_tensor_intarrayref(void* self, void* dims) {}
void lantern_numpy_t_tensor(void* self) {}
void lantern_pixel_shuffle_tensor_intt(void* self, void* upscale_factor) {}
void lantern_is_pinned_tensor(void* self) {}
void lantern_pin_memory_tensor(void* self) {}
void lantern_pinverse_tensor_double(void* self, void* rcond) {}
void lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(void* input, void* target, void* log_input, void* full, void* eps, void* reduction) {}
void lantern_scalar_tensor_scalar_tensoroptions(void* s, void* options) {}
void lantern_rand_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options) {}
void lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options) {}
void lantern_rand_intarrayref_tensoroptions(void* size, void* options) {}
void lantern_rand_intarrayref_generator_tensoroptions(void* size, void* generator, void* options) {}
void lantern_rand_out_tensor_intarrayref(void* out, void* size) {}
void lantern_rand_out_tensor_intarrayref_generator(void* out, void* size, void* generator) {}
void lantern_rand_like_tensor(void* self) {}
void lantern_rand_like_tensor_tensoroptions(void* self, void* options) {}
void lantern_randint_intt_intarrayref_tensoroptions(void* high, void* size, void* options) {}
void lantern_randint_intt_intarrayref_generator_tensoroptions(void* high, void* size, void* generator, void* options) {}
void lantern_randint_intt_intt_intarrayref_tensoroptions(void* low, void* high, void* size, void* options) {}
void lantern_randint_intt_intt_intarrayref_generator_tensoroptions(void* low, void* high, void* size, void* generator, void* options) {}
void lantern_randint_out_tensor_intt_intarrayref(void* out, void* high, void* size) {}
void lantern_randint_out_tensor_intt_intarrayref_generator(void* out, void* high, void* size, void* generator) {}
void lantern_randint_out_tensor_intt_intt_intarrayref(void* out, void* low, void* high, void* size) {}
void lantern_randint_out_tensor_intt_intt_intarrayref_generator(void* out, void* low, void* high, void* size, void* generator) {}
void lantern_randint_like_tensor_intt(void* self, void* high) {}
void lantern_randint_like_tensor_intt_intt(void* self, void* low, void* high) {}
void lantern_randint_like_tensor_intt_tensoroptions(void* self, void* high, void* options) {}
void lantern_randint_like_tensor_intt_intt_tensoroptions(void* self, void* low, void* high, void* options) {}
void lantern_randn_intarrayref_tensoroptions(void* size, void* options) {}
void lantern_randn_intarrayref_generator_tensoroptions(void* size, void* generator, void* options) {}
void lantern_randn_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options) {}
void lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options) {}
void lantern_randn_out_tensor_intarrayref(void* out, void* size) {}
void lantern_randn_out_tensor_intarrayref_generator(void* out, void* size, void* generator) {}
void lantern_randn_like_tensor(void* self) {}
void lantern_randn_like_tensor_tensoroptions(void* self, void* options) {}
void lantern_randperm_intt_tensoroptions(void* n, void* options) {}
void lantern_randperm_intt_generator_tensoroptions(void* n, void* generator, void* options) {}
void lantern_randperm_out_tensor_intt(void* out, void* n) {}
void lantern_randperm_out_tensor_intt_generator(void* out, void* n, void* generator) {}
void lantern_range_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options) {}
void lantern_range_scalar_scalar_tensoroptions(void* start, void* end, void* options) {}
void lantern_range_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step) {}
void lantern_reciprocal_tensor(void* self) {}
void lantern_reciprocal__tensor(void* self) {}
void lantern_reciprocal_out_tensor_tensor(void* out, void* self) {}
void lantern_neg_tensor(void* self) {}
void lantern_neg__tensor(void* self) {}
void lantern_neg_out_tensor_tensor(void* out, void* self) {}
void lantern_repeat_tensor_intarrayref(void* self, void* repeats) {}
void lantern_repeat_interleave_tensor(void* repeats) {}
void lantern_repeat_interleave_tensor_tensor_intt(void* self, void* repeats, void* dim) {}
void lantern_repeat_interleave_tensor_intt_intt(void* self, void* repeats, void* dim) {}
void lantern_reshape_tensor_intarrayref(void* self, void* shape) {}
void lantern__mkldnn_reshape_tensor_intarrayref(void* self, void* shape) {}
void lantern_reshape_as_tensor_tensor(void* self, void* other) {}
void lantern_round_tensor(void* self) {}
void lantern_round__tensor(void* self) {}
void lantern_round_out_tensor_tensor(void* out, void* self) {}
void lantern_rrelu_tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator) {}
void lantern_rrelu__tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator) {}
void lantern_relu_tensor(void* self) {}
void lantern_relu__tensor(void* self) {}
void lantern_prelu_tensor_tensor(void* self, void* weight) {}
void lantern_prelu_backward_tensor_tensor_tensor(void* grad_output, void* self, void* weight) {}
void lantern_gelu_tensor(void* self) {}
void lantern_gelu_backward_tensor_tensor(void* grad, void* self) {}
void lantern_hardshrink_tensor_scalar(void* self, void* lambd) {}
void lantern_hardshrink_backward_tensor_tensor_scalar(void* grad_out, void* self, void* lambd) {}
void lantern_rsqrt_tensor(void* self) {}
void lantern_rsqrt__tensor(void* self) {}
void lantern_rsqrt_out_tensor_tensor(void* out, void* self) {}
void lantern_select_tensor_dimname_intt(void* self, void* dim, void* index) {}
void lantern_select_tensor_intt_intt(void* self, void* dim, void* index) {}
void lantern_selu_tensor(void* self) {}
void lantern_selu__tensor(void* self) {}
void lantern_celu_tensor_scalar(void* self, void* alpha) {}
void lantern_celu__tensor_scalar(void* self, void* alpha) {}
void lantern_sigmoid_tensor(void* self) {}
void lantern_sigmoid__tensor(void* self) {}
void lantern_sigmoid_out_tensor_tensor(void* out, void* self) {}
void lantern_sin_tensor(void* self) {}
void lantern_sin__tensor(void* self) {}
void lantern_sin_out_tensor_tensor(void* out, void* self) {}
void lantern_sinh_tensor(void* self) {}
void lantern_sinh__tensor(void* self) {}
void lantern_sinh_out_tensor_tensor(void* out, void* self) {}
void lantern_detach_tensor(void* self) {}
void lantern_detach__tensor(void* self) {}
void lantern_size_tensor_intt(void* self, void* dim) {}
void lantern_size_tensor_dimname(void* self, void* dim) {}
void lantern_slice_tensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step) {}
void lantern_slogdet_tensor(void* self) {}
void lantern_smm_tensor_tensor(void* self, void* mat2) {}
void lantern_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype) {}
void lantern_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype) {}
void lantern__softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float) {}
void lantern__softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self) {}
void lantern_split_tensor_intt_intt(void* self, void* split_size, void* dim) {}
void lantern_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim) {}
void lantern_squeeze_tensor(void* self) {}
void lantern_squeeze_tensor_intt(void* self, void* dim) {}
void lantern_squeeze_tensor_dimname(void* self, void* dim) {}
void lantern_squeeze__tensor(void* self) {}
void lantern_squeeze__tensor_intt(void* self, void* dim) {}
void lantern_squeeze__tensor_dimname(void* self, void* dim) {}
void lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha) {}
void lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha) {}
void lantern_stack_tensorlist_intt(void* tensors, void* dim) {}
void lantern_stack_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim) {}
void lantern_stft_tensor_intt_intt_intt_tensor_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided) {}
void lantern_stride_tensor_intt(void* self, void* dim) {}
void lantern_stride_tensor_dimname(void* self, void* dim) {}
void lantern_sum_tensor_scalartype(void* self, void* dtype) {}
void lantern_sum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_sum_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_sum_to_size_tensor_intarrayref(void* self, void* size) {}
void lantern_sqrt_tensor(void* self) {}
void lantern_sqrt__tensor(void* self) {}
void lantern_sqrt_out_tensor_tensor(void* out, void* self) {}
void lantern_std_tensor_bool(void* self, void* unbiased) {}
void lantern_std_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_std_mean_tensor_bool(void* self, void* unbiased) {}
void lantern_std_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_std_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_std_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_std_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_std_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_prod_tensor_scalartype(void* self, void* dtype) {}
void lantern_prod_tensor_intt_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_prod_out_tensor_tensor_intt_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_prod_tensor_dimname_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_prod_out_tensor_tensor_dimname_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype) {}
void lantern_t_tensor(void* self) {}
void lantern_t__tensor(void* self) {}
void lantern_tan_tensor(void* self) {}
void lantern_tan__tensor(void* self) {}
void lantern_tan_out_tensor_tensor(void* out, void* self) {}
void lantern_tanh_tensor(void* self) {}
void lantern_tanh__tensor(void* self) {}
void lantern_tanh_out_tensor_tensor(void* out, void* self) {}
void lantern_tensordot_tensor_tensor_intarrayref_intarrayref(void* self, void* other, void* dims_self, void* dims_other) {}
void lantern_threshold_tensor_scalar_scalar(void* self, void* threshold, void* value) {}
void lantern_threshold__tensor_scalar_scalar(void* self, void* threshold, void* value) {}
void lantern_threshold_out_tensor_tensor_scalar_scalar(void* out, void* self, void* threshold, void* value) {}
void lantern_threshold_backward_tensor_tensor_scalar(void* grad_output, void* self, void* threshold) {}
void lantern_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1) {}
void lantern_transpose_tensor_dimname_dimname(void* self, void* dim0, void* dim1) {}
void lantern__mkldnn_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1) {}
void lantern_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1) {}
void lantern__mkldnn_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1) {}
void lantern_one_hot_tensor_intt(void* self, void* num_classes) {}
void lantern_flip_tensor_intarrayref(void* self, void* dims) {}
void lantern_roll_tensor_intarrayref_intarrayref(void* self, void* shifts, void* dims) {}
void lantern_rot90_tensor_intt_intarrayref(void* self, void* k, void* dims) {}
void lantern_trapz_tensor_tensor_intt(void* y, void* x, void* dim) {}
void lantern_trapz_tensor_double_intt(void* y, void* dx, void* dim) {}
void lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim) {}
void lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction) {}
void lantern_trunc_tensor(void* self) {}
void lantern_trunc__tensor(void* self) {}
void lantern_trunc_out_tensor_tensor(void* out, void* self) {}
void lantern_type_as_tensor_tensor(void* self, void* other) {}
void lantern__has_compatible_shallow_copy_type_tensor_tensor(void* self, void* from) {}
void lantern__unique_tensor_bool_bool(void* self, void* sorted, void* return_inverse) {}
void lantern_unique_dim_tensor_intt_bool_bool_bool(void* self, void* dim, void* sorted, void* return_inverse, void* return_counts) {}
void lantern_unique_consecutive_tensor_bool_bool_intt(void* self, void* return_inverse, void* return_counts, void* dim) {}
void lantern_unique_dim_consecutive_tensor_intt_bool_bool(void* self, void* dim, void* return_inverse, void* return_counts) {}
void lantern__unique2_tensor_bool_bool_bool(void* self, void* sorted, void* return_inverse, void* return_counts) {}
void lantern__unsafe_view_tensor_intarrayref(void* self, void* size) {}
void lantern_unsqueeze_tensor_intt(void* self, void* dim) {}
void lantern_unsqueeze__tensor_intt(void* self, void* dim) {}
void lantern_var_tensor_bool(void* self, void* unbiased) {}
void lantern_var_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_var_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_var_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_var_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_var_mean_tensor_bool(void* self, void* unbiased) {}
void lantern_var_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_var_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim) {}
void lantern_view_as_tensor_tensor(void* self, void* other) {}
void lantern_where_tensor_tensor_tensor(void* condition, void* self, void* other) {}
void lantern_where_tensor(void* condition) {}
void lantern__s_where_tensor_tensor_tensor(void* condition, void* self, void* other) {}
void lantern_norm_except_dim_tensor_intt_intt(void* v, void* pow, void* dim) {}
void lantern__weight_norm_tensor_tensor_intt(void* v, void* g, void* dim) {}
void lantern__weight_norm_cuda_interface_tensor_tensor_intt(void* v, void* g, void* dim) {}
void lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim) {}
void lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim) {}
void lantern_zeros_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options) {}
void lantern_zeros_intarrayref_tensoroptions(void* size, void* options) {}
void lantern_zeros_out_tensor_intarrayref(void* out, void* size) {}
void lantern_zeros_like_tensor(void* self) {}
void lantern_zeros_like_tensor_tensoroptions(void* self, void* options) {}
void lantern__standard_gamma_grad_tensor_tensor(void* self, void* output) {}
void lantern__standard_gamma_tensor_generator(void* self, void* generator) {}
void lantern__dirichlet_grad_tensor_tensor_tensor(void* x, void* alpha, void* total) {}
void lantern__sample_dirichlet_tensor_generator(void* self, void* generator) {}
void lantern_poisson_tensor_generator(void* self, void* generator) {}
void lantern_native_norm_tensor_scalar(void* self, void* p) {}
void lantern__sparse_sum_tensor(void* self) {}
void lantern__sparse_sum_tensor_scalartype(void* self, void* dtype) {}
void lantern__sparse_sum_tensor_intarrayref(void* self, void* dim) {}
void lantern__sparse_sum_tensor_intarrayref_scalartype(void* self, void* dim, void* dtype) {}
void lantern__sparse_sum_backward_tensor_tensor_intarrayref(void* grad, void* self, void* dim) {}
void lantern_norm_tensor_scalar_scalartype(void* self, void* p, void* dtype) {}
void lantern_norm_tensor_scalar(void* self, void* p) {}
void lantern_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype) {}
void lantern_norm_tensor_scalar_intarrayref_bool(void* self, void* p, void* dim, void* keepdim) {}
void lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype) {}
void lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(void* out, void* self, void* p, void* dim, void* keepdim) {}
void lantern_norm_tensor_scalar_dimnamelist_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype) {}
void lantern_norm_tensor_scalar_dimnamelist_bool(void* self, void* p, void* dim, void* keepdim) {}
void lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype) {}
void lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(void* out, void* self, void* p, void* dim, void* keepdim) {}
void lantern_frobenius_norm_tensor(void* self) {}
void lantern_frobenius_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim) {}
void lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_nuclear_norm_tensor_bool(void* self, void* keepdim) {}
void lantern_nuclear_norm_out_tensor_tensor_bool(void* out, void* self, void* keepdim) {}
void lantern_nuclear_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim) {}
void lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim) {}
void lantern_clone_tensor(void* self) {}
void lantern_resize_as__tensor_tensor(void* self, void* the_template) {}
void lantern_pow_out_tensor_tensor_scalar(void* out, void* self, void* exponent) {}
void lantern_pow_tensor_scalar(void* self, void* exponent) {}
void lantern_zero__tensor(void* self) {}
void lantern_sub_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha) {}
void lantern_sub_tensor_tensor_scalar(void* self, void* other, void* alpha) {}
void lantern_sub__tensor_tensor_scalar(void* self, void* other, void* alpha) {}
void lantern_sub_tensor_scalar_scalar(void* self, void* other, void* alpha) {}
void lantern_sub__tensor_scalar_scalar(void* self, void* other, void* alpha) {}
void lantern_rsub_tensor_tensor_scalar(void* self, void* other, void* alpha) {}
void lantern_rsub_tensor_scalar_scalar(void* self, void* other, void* alpha) {}
void lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* sparse, void* dense, void* beta, void* alpha) {}
void lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha) {}
void lantern_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha) {}
void lantern_addmm__tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha) {}
void lantern_sparse_coo_tensor_intarrayref_tensoroptions(void* size, void* options) {}
void lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(void* indices, void* values, void* options) {}
void lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options) {}
void lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options) {}
void lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* options) {}
void lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* indices, void* values, void* options) {}
void lantern_sparse_resize__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim) {}
void lantern_sparse_resize_and_clear__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim) {}
void lantern_sparse_mask_tensor_tensor(void* self, void* mask) {}
void lantern_to_dense_tensor(void* self) {}
void lantern_to_dense_backward_tensor_tensor(void* grad, void* input) {}
void lantern_sparse_dim_tensor(void* self) {}
void lantern__dimi_tensor(void* self) {}
void lantern_dense_dim_tensor(void* self) {}
void lantern__dimv_tensor(void* self) {}
void lantern__nnz_tensor(void* self) {}
void lantern_coalesce_tensor(void* self) {}
void lantern_is_coalesced_tensor(void* self) {}
void lantern__indices_tensor(void* self) {}
void lantern__values_tensor(void* self) {}
void lantern__coalesced__tensor_bool(void* self, void* coalesced) {}
void lantern_indices_tensor(void* self) {}
void lantern_values_tensor(void* self) {}
void lantern_hspmm_out_tensor_tensor_tensor(void* out, void* mat1, void* mat2) {}
void lantern_hspmm_tensor_tensor(void* mat1, void* mat2) {}
void lantern_copy_sparse_to_sparse__tensor_tensor_bool(void* self, void* src, void* non_blocking) {}
void lantern_numel_tensor(void* self) {}
void lantern_unbind_tensor_intt(void* self, void* dim) {}
void lantern_unbind_tensor_dimname(void* self, void* dim) {}
void lantern_to_sparse_tensor_intt(void* self, void* sparse_dim) {}
void lantern_to_sparse_tensor(void* self) {}
void lantern_to_mkldnn_tensor(void* self) {}
void lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups) {}
void lantern_to_mkldnn_backward_tensor_tensor(void* grad, void* input) {}
void lantern_quantize_per_tensor_tensor_double_intt_scalartype(void* self, void* scale, void* zero_point, void* dtype) {}
void lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype(void* self, void* scales, void* zero_points, void* axis, void* dtype) {}
void lantern_dequantize_tensor(void* self) {}
void lantern_q_scale_tensor(void* self) {}
void lantern_q_zero_point_tensor(void* self) {}
void lantern_q_per_channel_scales_tensor(void* self) {}
void lantern_q_per_channel_zero_points_tensor(void* self) {}
void lantern_q_per_channel_axis_tensor(void* self) {}
void lantern_int_repr_tensor(void* self) {}
void lantern__make_per_tensor_quantized_tensor_tensor_double_intt(void* self, void* scale, void* zero_point) {}
void lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt(void* self, void* scale, void* zero_point, void* axis) {}
void lantern_qscheme_tensor(void* self) {}
void lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max) {}
void lantern_fake_quantize_per_tensor_affine_backward_tensor_tensor_double_intt_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max) {}
void lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max) {}
void lantern_fake_quantize_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max) {}
void lantern_to_tensor_tensoroptions_bool_bool(void* self, void* options, void* non_blocking, void* copy) {}
void lantern_to_tensor_device_scalartype_bool_bool(void* self, void* device, void* dtype, void* non_blocking, void* copy) {}
void lantern_to_tensor_scalartype_bool_bool(void* self, void* dtype, void* non_blocking, void* copy) {}
void lantern_to_tensor_tensor_bool_bool(void* self, void* other, void* non_blocking, void* copy) {}
void lantern_meshgrid_tensorlist(void* tensors) {}
void lantern_cartesian_prod_tensorlist(void* tensors) {}
void lantern_combinations_tensor_intt_bool(void* self, void* r, void* with_replacement) {}
void lantern_item_tensor(void* self) {}
void lantern_result_type_tensor_tensor(void* tensor, void* other) {}
void lantern_result_type_tensor_scalar(void* tensor, void* other) {}
void lantern_result_type_scalar_tensor(void* scalar, void* tensor) {}
void lantern_result_type_scalar_scalar(void* scalar1, void* scalar2) {}
void lantern_can_cast_scalartype_scalartype(void* from, void* to) {}
void lantern_promote_types_scalartype_scalartype(void* type1, void* type2) {}
void lantern__local_scalar_dense_tensor(void* self) {}
void lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* cx, void* input_bias, void* hidden_bias) {}
void lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool(void* grad_hy, void* grad_cy, void* cx, void* cy, void* workspace, void* has_bias) {}
void lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* grad_cy, void* input_gates, void* hidden_gates, void* input_bias, void* hidden_bias, void* cx, void* cy) {}
void lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias) {}
void lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool(void* grad_hy, void* workspace, void* has_bias) {}
void lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias) {}
void lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first) {}
void lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional) {}
void lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first) {}
void lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional) {}
void lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first) {}
void lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional) {}
void lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first) {}
void lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional) {}
void lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh) {}
void lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh) {}
void lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh) {}
void lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh) {}
void lantern_quantized_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool_scalartype_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first, void* dtype, void* use_dynamic) {}
void lantern_quantized_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first) {}
void lantern_quantized_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional) {}
void lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh) {}
void lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh) {}
void lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh) {}
void lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh) {}
void lantern__pack_padded_sequence_tensor_tensor_bool(void* input, void* lengths, void* batch_first) {}
void lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(void* grad, void* input_size, void* batch_sizes, void* batch_first) {}
void lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt(void* data, void* batch_sizes, void* batch_first, void* padding_value, void* total_length) {}
void lantern_set__tensor_storage(void* self, void* source) {}
void lantern_set__tensor_storage_intt_intarrayref_intarrayref(void* self, void* source, void* storage_offset, void* size, void* stride) {}
void lantern_set__tensor_tensor(void* self, void* source) {}
void lantern_set__tensor(void* self) {}
void lantern_set_quantizer__tensor_constquantizerptr(void* self, void* quantizer) {}
void lantern_is_set_to_tensor_tensor(void* self, void* tensor) {}
void lantern_masked_fill__tensor_tensor_scalar(void* self, void* mask, void* value) {}
void lantern_masked_fill_tensor_tensor_scalar(void* self, void* mask, void* value) {}
void lantern_masked_fill__tensor_tensor_tensor(void* self, void* mask, void* value) {}
void lantern_masked_fill_tensor_tensor_tensor(void* self, void* mask, void* value) {}
void lantern_masked_scatter__tensor_tensor_tensor(void* self, void* mask, void* source) {}
void lantern_masked_scatter_tensor_tensor_tensor(void* self, void* mask, void* source) {}
void lantern_view_tensor_intarrayref(void* self, void* size) {}
void lantern_put__tensor_tensor_tensor_bool(void* self, void* index, void* source, void* accumulate) {}
void lantern_index_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern_index_fill__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill__tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_index_fill_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value) {}
void lantern_scatter__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_scatter_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_scatter__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_scatter_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_scatter_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_scatter_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value) {}
void lantern_scatter_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_scatter_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_scatter_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src) {}
void lantern_lt__tensor_scalar(void* self, void* other) {}
void lantern_lt__tensor_tensor(void* self, void* other) {}
void lantern_gt__tensor_scalar(void* self, void* other) {}
void lantern_gt__tensor_tensor(void* self, void* other) {}
void lantern_le__tensor_scalar(void* self, void* other) {}
void lantern_le__tensor_tensor(void* self, void* other) {}
void lantern_ge__tensor_scalar(void* self, void* other) {}
void lantern_ge__tensor_tensor(void* self, void* other) {}
void lantern_eq__tensor_scalar(void* self, void* other) {}
void lantern_eq__tensor_tensor(void* self, void* other) {}
void lantern_ne__tensor_scalar(void* self, void* other) {}
void lantern_ne__tensor_tensor(void* self, void* other) {}
void lantern___and___tensor_scalar(void* self, void* other) {}
void lantern___and___tensor_tensor(void* self, void* other) {}
void lantern___iand___tensor_scalar(void* self, void* other) {}
void lantern___iand___tensor_tensor(void* self, void* other) {}
void lantern___or___tensor_scalar(void* self, void* other) {}
void lantern___or___tensor_tensor(void* self, void* other) {}
void lantern___ior___tensor_scalar(void* self, void* other) {}
void lantern___ior___tensor_tensor(void* self, void* other) {}
void lantern___xor___tensor_scalar(void* self, void* other) {}
void lantern___xor___tensor_tensor(void* self, void* other) {}
void lantern___ixor___tensor_scalar(void* self, void* other) {}
void lantern___ixor___tensor_tensor(void* self, void* other) {}
void lantern___lshift___tensor_scalar(void* self, void* other) {}
void lantern___lshift___tensor_tensor(void* self, void* other) {}
void lantern___ilshift___tensor_scalar(void* self, void* other) {}
void lantern___ilshift___tensor_tensor(void* self, void* other) {}
void lantern___rshift___tensor_scalar(void* self, void* other) {}
void lantern___rshift___tensor_tensor(void* self, void* other) {}
void lantern___irshift___tensor_scalar(void* self, void* other) {}
void lantern___irshift___tensor_tensor(void* self, void* other) {}
void lantern_lgamma__tensor(void* self) {}
void lantern_atan2__tensor_tensor(void* self, void* other) {}
void lantern_tril__tensor_intt(void* self, void* diagonal) {}
void lantern_triu__tensor_intt(void* self, void* diagonal) {}
void lantern_digamma__tensor(void* self) {}
void lantern_polygamma__tensor_intt(void* self, void* n) {}
void lantern_renorm__tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm) {}
void lantern_pow__tensor_scalar(void* self, void* exponent) {}
void lantern_pow__tensor_tensor(void* self, void* exponent) {}
void lantern_lerp__tensor_tensor_scalar(void* self, void* end, void* weight) {}
void lantern_lerp__tensor_tensor_tensor(void* self, void* end, void* weight) {}
void lantern_fmod__tensor_scalar(void* self, void* other) {}
void lantern_fmod__tensor_tensor(void* self, void* other) {}
void lantern_remainder__tensor_scalar(void* self, void* other) {}
void lantern_remainder__tensor_tensor(void* self, void* other) {}
void lantern_addbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_addbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha) {}
void lantern_addcdiv__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_random__tensor_intt_intt_generator(void* self, void* from, void* to, void* generator) {}
void lantern_random__tensor_intt_generator(void* self, void* to, void* generator) {}
void lantern_random__tensor_generator(void* self, void* generator) {}
void lantern_uniform__tensor_double_double_generator(void* self, void* from, void* to, void* generator) {}
void lantern_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator) {}
void lantern_cauchy__tensor_double_double_generator(void* self, void* median, void* sigma, void* generator) {}
void lantern_log_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator) {}
void lantern_exponential__tensor_double_generator(void* self, void* lambd, void* generator) {}
void lantern_geometric__tensor_double_generator(void* self, void* p, void* generator) {}
void lantern_diag_out_tensor_tensor_intt(void* out, void* self, void* diagonal) {}
void lantern_diag_tensor_intt(void* self, void* diagonal) {}
void lantern_cross_out_tensor_tensor_tensor_intt(void* out, void* self, void* other, void* dim) {}
void lantern_cross_tensor_tensor_intt(void* self, void* other, void* dim) {}
void lantern_triu_out_tensor_tensor_intt(void* out, void* self, void* diagonal) {}
void lantern_triu_tensor_intt(void* self, void* diagonal) {}
void lantern_tril_out_tensor_tensor_intt(void* out, void* self, void* diagonal) {}
void lantern_tril_tensor_intt(void* self, void* diagonal) {}
void lantern_tril_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options) {}
void lantern_triu_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options) {}
void lantern_trace_tensor(void* self) {}
void lantern_ne_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_ne_tensor_scalar(void* self, void* other) {}
void lantern_ne_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_ne_tensor_tensor(void* self, void* other) {}
void lantern_eq_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_eq_tensor_scalar(void* self, void* other) {}
void lantern_eq_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_eq_tensor_tensor(void* self, void* other) {}
void lantern_ge_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_ge_tensor_scalar(void* self, void* other) {}
void lantern_ge_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_ge_tensor_tensor(void* self, void* other) {}
void lantern_le_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_le_tensor_scalar(void* self, void* other) {}
void lantern_le_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_le_tensor_tensor(void* self, void* other) {}
void lantern_gt_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_gt_tensor_scalar(void* self, void* other) {}
void lantern_gt_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_gt_tensor_tensor(void* self, void* other) {}
void lantern_lt_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_lt_tensor_scalar(void* self, void* other) {}
void lantern_lt_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_lt_tensor_tensor(void* self, void* other) {}
void lantern_take_out_tensor_tensor_tensor(void* out, void* self, void* index) {}
void lantern_take_tensor_tensor(void* self, void* index) {}
void lantern_index_select_out_tensor_tensor_intt_tensor(void* out, void* self, void* dim, void* index) {}
void lantern_index_select_tensor_intt_tensor(void* self, void* dim, void* index) {}
void lantern_index_select_out_tensor_tensor_dimname_tensor(void* out, void* self, void* dim, void* index) {}
void lantern_index_select_tensor_dimname_tensor(void* self, void* dim, void* index) {}
void lantern_masked_select_out_tensor_tensor_tensor(void* out, void* self, void* mask) {}
void lantern_masked_select_tensor_tensor(void* self, void* mask) {}
void lantern_nonzero_out_tensor_tensor(void* out, void* self) {}
void lantern_nonzero_tensor(void* self) {}
void lantern_nonzero_numpy_tensor(void* self) {}
void lantern_gather_out_tensor_tensor_intt_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad) {}
void lantern_gather_tensor_intt_tensor_bool(void* self, void* dim, void* index, void* sparse_grad) {}
void lantern_gather_out_tensor_tensor_dimname_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad) {}
void lantern_gather_tensor_dimname_tensor_bool(void* self, void* dim, void* index, void* sparse_grad) {}
void lantern__gather_sparse_backward_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* grad) {}
void lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_addcmul_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_addcmul__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_addcdiv_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value) {}
void lantern_lstsq_out_tensor_tensor_tensor_tensor(void* X, void* qr, void* self, void* A) {}
void lantern_lstsq_tensor_tensor(void* self, void* A) {}
void lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool(void* X, void* M, void* self, void* A, void* upper, void* transpose, void* unitriangular) {}
void lantern_triangular_solve_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular) {}
void lantern__triangular_solve_helper_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular) {}
void lantern_symeig_out_tensor_tensor_tensor_bool_bool(void* e, void* V, void* self, void* eigenvectors, void* upper) {}
void lantern_symeig_tensor_bool_bool(void* self, void* eigenvectors, void* upper) {}
void lantern__symeig_helper_tensor_bool_bool(void* self, void* eigenvectors, void* upper) {}
void lantern_eig_out_tensor_tensor_tensor_bool(void* e, void* v, void* self, void* eigenvectors) {}
void lantern_eig_tensor_bool(void* self, void* eigenvectors) {}
void lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool(void* U, void* S, void* V, void* self, void* some, void* compute_uv) {}
void lantern_svd_tensor_bool_bool(void* self, void* some, void* compute_uv) {}
void lantern__svd_helper_tensor_bool_bool(void* self, void* some, void* compute_uv) {}
void lantern_cholesky_out_tensor_tensor_bool(void* out, void* self, void* upper) {}
void lantern_cholesky_tensor_bool(void* self, void* upper) {}
void lantern__cholesky_helper_tensor_bool(void* self, void* upper) {}
void lantern_cholesky_solve_out_tensor_tensor_tensor_bool(void* out, void* self, void* input2, void* upper) {}
void lantern_cholesky_solve_tensor_tensor_bool(void* self, void* input2, void* upper) {}
void lantern__cholesky_solve_helper_tensor_tensor_bool(void* self, void* A, void* upper) {}
void lantern_solve_tensor_tensor(void* self, void* A) {}
void lantern_solve_out_tensor_tensor_tensor_tensor(void* solution, void* lu, void* self, void* A) {}
void lantern__solve_helper_tensor_tensor(void* self, void* A) {}
void lantern_cholesky_inverse_out_tensor_tensor_bool(void* out, void* self, void* upper) {}
void lantern_cholesky_inverse_tensor_bool(void* self, void* upper) {}
void lantern_qr_out_tensor_tensor_tensor_bool(void* Q, void* R, void* self, void* some) {}
void lantern_qr_tensor_bool(void* self, void* some) {}
void lantern__qr_helper_tensor_bool(void* self, void* some) {}
void lantern_geqrf_out_tensor_tensor_tensor(void* a, void* tau, void* self) {}
void lantern_geqrf_tensor(void* self) {}
void lantern_orgqr_out_tensor_tensor_tensor(void* out, void* self, void* input2) {}
void lantern_orgqr_tensor_tensor(void* self, void* input2) {}
void lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool(void* out, void* self, void* input2, void* input3, void* left, void* transpose) {}
void lantern_ormqr_tensor_tensor_tensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose) {}
void lantern__lu_with_info_tensor_bool_bool(void* self, void* pivot, void* check_errors) {}
void lantern_lu_solve_out_tensor_tensor_tensor_tensor(void* out, void* self, void* LU_data, void* LU_pivots) {}
void lantern_lu_solve_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots) {}
void lantern__lu_solve_helper_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots) {}
void lantern_multinomial_out_tensor_tensor_intt_bool_generator(void* out, void* self, void* num_samples, void* replacement, void* generator) {}
void lantern_multinomial_tensor_intt_bool_generator(void* self, void* num_samples, void* replacement, void* generator) {}
void lantern__multinomial_alias_setup_tensor(void* probs) {}
void lantern__multinomial_alias_draw_tensor_tensor_intt_generator(void* J, void* q, void* num_samples, void* generator) {}
void lantern_lgamma_out_tensor_tensor(void* out, void* self) {}
void lantern_lgamma_tensor(void* self) {}
void lantern_digamma_out_tensor_tensor(void* out, void* self) {}
void lantern_digamma_tensor(void* self) {}
void lantern_polygamma_out_tensor_intt_tensor(void* out, void* n, void* self) {}
void lantern_polygamma_intt_tensor(void* n, void* self) {}
void lantern_erfinv_tensor(void* self) {}
void lantern_erfinv__tensor(void* self) {}
void lantern_erfinv_out_tensor_tensor(void* out, void* self) {}
void lantern_sign_tensor(void* self) {}
void lantern_sign__tensor(void* self) {}
void lantern_sign_out_tensor_tensor(void* out, void* self) {}
void lantern_dist_tensor_tensor_scalar(void* self, void* other, void* p) {}
void lantern_atan2_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_atan2_tensor_tensor(void* self, void* other) {}
void lantern_lerp_out_tensor_tensor_tensor_scalar(void* out, void* self, void* end, void* weight) {}
void lantern_lerp_out_tensor_tensor_tensor_tensor(void* out, void* self, void* end, void* weight) {}
void lantern_lerp_tensor_tensor_scalar(void* self, void* end, void* weight) {}
void lantern_lerp_tensor_tensor_tensor(void* self, void* end, void* weight) {}
void lantern_histc_out_tensor_tensor_intt_scalar_scalar(void* out, void* self, void* bins, void* min, void* max) {}
void lantern_histc_tensor_intt_scalar_scalar(void* self, void* bins, void* min, void* max) {}
void lantern_fmod_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_fmod_tensor_scalar(void* self, void* other) {}
void lantern_fmod_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_fmod_tensor_tensor(void* self, void* other) {}
void lantern_remainder_out_tensor_tensor_scalar(void* out, void* self, void* other) {}
void lantern_remainder_tensor_scalar(void* self, void* other) {}
void lantern_remainder_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_remainder_tensor_tensor(void* self, void* other) {}
void lantern_min_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_min_tensor_tensor(void* self, void* other) {}
void lantern_min_tensor(void* self) {}
void lantern_max_out_tensor_tensor_tensor(void* out, void* self, void* other) {}
void lantern_max_tensor_tensor(void* self, void* other) {}
void lantern_max_tensor(void* self) {}
void lantern_median_tensor(void* self) {}
void lantern_sort_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* descending) {}
void lantern_sort_tensor_intt_bool(void* self, void* dim, void* descending) {}
void lantern_sort_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* descending) {}
void lantern_sort_tensor_dimname_bool(void* self, void* dim, void* descending) {}
void lantern_argsort_tensor_intt_bool(void* self, void* dim, void* descending) {}
void lantern_argsort_tensor_dimname_bool(void* self, void* dim, void* descending) {}
void lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool(void* values, void* indices, void* self, void* k, void* dim, void* largest, void* sorted) {}
void lantern_topk_tensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted) {}
void lantern_all_tensor(void* self) {}
void lantern_any_tensor(void* self) {}
void lantern_renorm_out_tensor_tensor_scalar_intt_scalar(void* out, void* self, void* p, void* dim, void* maxnorm) {}
void lantern_renorm_tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm) {}
void lantern_unfold_tensor_intt_intt_intt(void* self, void* dimension, void* size, void* step) {}
void lantern_equal_tensor_tensor(void* self, void* other) {}
void lantern_pow_out_tensor_tensor_tensor(void* out, void* self, void* exponent) {}
void lantern_pow_tensor_tensor(void* self, void* exponent) {}
void lantern_pow_out_tensor_scalar_tensor(void* out, void* self, void* exponent) {}
void lantern_pow_scalar_tensor(void* self, void* exponent) {}
void lantern_normal_out_tensor_tensor_double_generator(void* out, void* mean, void* std, void* generator) {}
void lantern_normal_tensor_double_generator(void* mean, void* std, void* generator) {}
void lantern_normal_out_tensor_double_tensor_generator(void* out, void* mean, void* std, void* generator) {}
void lantern_normal_double_tensor_generator(void* mean, void* std, void* generator) {}
void lantern_normal_out_tensor_tensor_tensor_generator(void* out, void* mean, void* std, void* generator) {}
void lantern_normal_tensor_tensor_generator(void* mean, void* std, void* generator) {}
void lantern_normal_double_double_intarrayref_generator_tensoroptions(void* mean, void* std, void* size, void* generator, void* options) {}
void lantern_normal_out_tensor_double_double_intarrayref_generator(void* out, void* mean, void* std, void* size, void* generator) {}
void lantern_alias_tensor(void* self) {}
void lantern__addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern__addr__tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern__addr_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha) {}
void lantern__index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source) {}
void lantern__cumsum_tensor_intt(void* self, void* dim) {}
void lantern__cumsum_out_tensor_tensor_intt(void* out, void* self, void* dim) {}
void lantern__cumprod_tensor_intt(void* self, void* dim) {}
void lantern__cumprod_out_tensor_tensor_intt(void* out, void* self, void* dim) {}
void lantern__var_tensor_bool(void* self, void* unbiased) {}
void lantern__std_tensor_bool(void* self, void* unbiased) {}
void lantern__cat_tensorlist_intt(void* tensors, void* dim) {}
void lantern__cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim) {}
void lantern__mode_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern__mode_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim) {}
void lantern__max_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern__max_out_tensor_tensor_tensor_intt_bool(void* max, void* max_indices, void* self, void* dim, void* keepdim) {}
void lantern__min_tensor_intt_bool(void* self, void* dim, void* keepdim) {}
void lantern__min_out_tensor_tensor_tensor_intt_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim) {}
void lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* weight, void* reduction) {}
void lantern_binary_cross_entropy_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* reduction) {}
void lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction) {}
void lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* reduction) {}
void lantern_mse_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction) {}
void lantern_mse_loss_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction) {}
void lantern_mse_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction) {}
void lantern_l1_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction) {}
void lantern_l1_loss_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction) {}
void lantern_l1_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction) {}
void lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* out, void* self, void* target, void* p, void* margin, void* weight, void* reduction) {}
void lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt(void* self, void* target, void* p, void* margin, void* weight, void* reduction) {}
void lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction) {}
void lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction) {}
void lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction) {}
void lantern_multilabel_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt(void* output, void* is_target, void* self, void* target, void* reduction) {}
void lantern_multilabel_margin_loss_forward_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* is_target) {}
void lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor(void* grad_output, void* self, void* target, void* reduction, void* is_target) {}
void lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight) {}
void lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight) {}
void lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index) {}
void lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight) {}
void lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight) {}
void lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction) {}
void lantern_smooth_l1_loss_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction) {}
void lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction) {}
void lantern_soft_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction) {}
void lantern_soft_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction) {}
void lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction) {}
void lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction) {}
void lantern_elu_out_tensor_tensor_scalar_scalar_scalar(void* out, void* self, void* alpha, void* scale, void* input_scale) {}
void lantern_elu_tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale) {}
void lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_tensor(void* grad_input, void* grad_output, void* alpha, void* scale, void* input_scale, void* output) {}
void lantern_elu_backward_tensor_scalar_scalar_scalar_tensor(void* grad_output, void* alpha, void* scale, void* input_scale, void* output) {}
void lantern_elu__tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale) {}
void lantern_glu_out_tensor_tensor_intt(void* out, void* self, void* dim) {}
void lantern_glu_tensor_intt(void* self, void* dim) {}
void lantern_glu_backward_out_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* dim) {}
void lantern_glu_backward_tensor_tensor_intt(void* grad_output, void* self, void* dim) {}
void lantern_hardtanh_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min_val, void* max_val) {}
void lantern_hardtanh_tensor_scalar_scalar(void* self, void* min_val, void* max_val) {}
void lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar(void* grad_input, void* grad_output, void* self, void* min_val, void* max_val) {}
void lantern_hardtanh_backward_tensor_tensor_scalar_scalar(void* grad_output, void* self, void* min_val, void* max_val) {}
void lantern_hardtanh__tensor_scalar_scalar(void* self, void* min_val, void* max_val) {}
void lantern_leaky_relu_out_tensor_tensor_scalar(void* out, void* self, void* negative_slope) {}
void lantern_leaky_relu_tensor_scalar(void* self, void* negative_slope) {}
void lantern_leaky_relu_backward_out_tensor_tensor_tensor_scalar(void* grad_input, void* grad_output, void* self, void* negative_slope) {}
void lantern_leaky_relu_backward_tensor_tensor_scalar(void* grad_output, void* self, void* negative_slope) {}
void lantern_leaky_relu__tensor_scalar(void* self, void* negative_slope) {}
void lantern_log_sigmoid_out_tensor_tensor(void* out, void* self) {}
void lantern_log_sigmoid_tensor(void* self) {}
void lantern_log_sigmoid_forward_out_tensor_tensor_tensor(void* output, void* buffer, void* self) {}
void lantern_log_sigmoid_forward_tensor(void* self) {}
void lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* buffer) {}
void lantern_log_sigmoid_backward_tensor_tensor_tensor(void* grad_output, void* self, void* buffer) {}
void lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator(void* out, void* self, void* noise, void* lower, void* upper, void* training, void* generator) {}
void lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator) {}
void lantern_rrelu_with_noise_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_bool(void* grad_input, void* grad_output, void* self, void* noise, void* lower, void* upper, void* training) {}
void lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool(void* grad_output, void* self, void* noise, void* lower, void* upper, void* training) {}
void lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator) {}
void lantern_softplus_out_tensor_tensor_scalar_scalar(void* out, void* self, void* beta, void* threshold) {}
void lantern_softplus_tensor_scalar_scalar(void* self, void* beta, void* threshold) {}
void lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor(void* grad_input, void* grad_output, void* self, void* beta, void* threshold, void* output) {}
void lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor(void* grad_output, void* self, void* beta, void* threshold, void* output) {}
void lantern_softshrink_out_tensor_tensor_scalar(void* out, void* self, void* lambd) {}
void lantern_softshrink_tensor_scalar(void* self, void* lambd) {}
void lantern_softshrink_backward_out_tensor_tensor_tensor_scalar(void* grad_input, void* grad_output, void* self, void* lambd) {}
void lantern_softshrink_backward_tensor_tensor_scalar(void* grad_output, void* self, void* lambd) {}
void lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size) {}
void lantern_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size) {}
void lantern__adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size) {}
void lantern__adaptive_avg_pool2d_backward_tensor_tensor(void* grad_output, void* self) {}
void lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size) {}
void lantern_adaptive_avg_pool3d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self) {}
void lantern_adaptive_avg_pool3d_backward_tensor_tensor(void* grad_output, void* self) {}
void lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size) {}
void lantern_adaptive_max_pool2d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices) {}
void lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices) {}
void lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size) {}
void lantern_adaptive_max_pool3d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices) {}
void lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices) {}
void lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override) {}
void lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples) {}
void lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples) {}
void lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices) {}
void lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices) {}
void lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples) {}
void lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples) {}
void lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices) {}
void lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices) {}
void lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices) {}
void lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices) {}
void lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode) {}
void lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices) {}
void lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices) {}
void lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* self, void* indices, void* output_size) {}
void lantern_max_unpool2d_tensor_tensor_intarrayref(void* self, void* indices, void* output_size) {}
void lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size) {}
void lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref(void* grad_output, void* self, void* indices, void* output_size) {}
void lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* indices, void* output_size, void* stride, void* padding) {}
void lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref(void* self, void* indices, void* output_size, void* stride, void* padding) {}
void lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding) {}
void lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding) {}
void lantern_reflection_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding) {}
void lantern_reflection_pad1d_tensor_intarrayref(void* self, void* padding) {}
void lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding) {}
void lantern_reflection_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding) {}
void lantern_reflection_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding) {}
void lantern_reflection_pad2d_tensor_intarrayref(void* self, void* padding) {}
void lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding) {}
void lantern_reflection_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding) {}
void lantern_replication_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding) {}
void lantern_replication_pad1d_tensor_intarrayref(void* self, void* padding) {}
void lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding) {}
void lantern_replication_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding) {}
void lantern_replication_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding) {}
void lantern_replication_pad2d_tensor_intarrayref(void* self, void* padding) {}
void lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding) {}
void lantern_replication_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding) {}
void lantern_replication_pad3d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding) {}
void lantern_replication_pad3d_tensor_intarrayref(void* self, void* padding) {}
void lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding) {}
void lantern_replication_pad3d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding) {}
void lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* output_size, void* align_corners) {}
void lantern_upsample_linear1d_tensor_intarrayref_bool(void* self, void* output_size, void* align_corners) {}
void lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool(void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* output_size, void* align_corners) {}
void lantern_upsample_bilinear2d_tensor_intarrayref_bool(void* self, void* output_size, void* align_corners) {}
void lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool(void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* output_size, void* align_corners) {}
void lantern_upsample_bicubic2d_tensor_intarrayref_bool(void* self, void* output_size, void* align_corners) {}
void lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool(void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* output_size, void* align_corners) {}
void lantern_upsample_trilinear3d_tensor_intarrayref_bool(void* self, void* output_size, void* align_corners) {}
void lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool(void* grad_output, void* output_size, void* input_size, void* align_corners) {}
void lantern_upsample_nearest1d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size) {}
void lantern_upsample_nearest1d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref(void* grad_input, void* grad_output, void* output_size, void* input_size) {}
void lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref(void* grad_output, void* output_size, void* input_size) {}
void lantern_upsample_nearest2d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size) {}
void lantern_upsample_nearest2d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref(void* grad_input, void* grad_output, void* output_size, void* input_size) {}
void lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref(void* grad_output, void* output_size, void* input_size) {}
void lantern_upsample_nearest3d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size) {}
void lantern_upsample_nearest3d_tensor_intarrayref(void* self, void* output_size) {}
void lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref(void* grad_input, void* grad_output, void* output_size, void* input_size) {}
void lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref(void* grad_output, void* output_size, void* input_size) {}
void lantern_sigmoid_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output) {}
void lantern_sigmoid_backward_tensor_tensor(void* grad_output, void* output) {}
void lantern_tanh_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output) {}
void lantern_tanh_backward_tensor_tensor(void* grad_output, void* output) {}
void lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation) {}
void lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation) {}
void lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones) {}
void lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones, void* output_mask) {}
void lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation) {}
void lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation) {}
void lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input) {}
void lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input, void* output_mask) {}
void lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input) {}
void lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask) {}
void lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_weight, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation) {}
void lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask) {}
void lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding) {}
void lantern_thnn_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input) {}
void lantern_thnn_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask) {}
void lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask) {}
void lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation) {}
void lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask) {}
void lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride) {}
void lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride) {}
/* Autogen Body -- End */
