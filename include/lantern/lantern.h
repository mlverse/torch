#ifndef __LANTERN_H__
#define __LANTERN_H__

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#ifdef LANTERN_BUILD
#define LANTERN_PTR
#define LANTERN_HEADERS_ONLY
#ifdef _WIN32
#define LANTERN_API extern "C" __declspec(dllexport)
#endif
#else
#define LANTERN_PTR *
#endif

#ifndef LANTERN_API
#ifdef LANTERN_HEADERS_ONLY
#define LANTERN_API extern
#else
#define LANTERN_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
LANTERN_API void (LANTERN_PTR lanternTest)();

/* Autogen Headers -- Start */
LANTERN_API void* (LANTERN_PTR lantern__cast_byte_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_char_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_double_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_float_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_int_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_long_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_short_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__cast_half_tensor_bool)(void* self, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern_backward_tensor_tensor_bool_bool)(void* self, void* gradient, void* keep_graph, void* create_graph);
LANTERN_API void* (LANTERN_PTR lantern_set_data_tensor_tensor)(void* self, void* new_data);
LANTERN_API void* (LANTERN_PTR lantern_data_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_leaf_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_output_nr_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__version_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_rename__tensor_dimnamelist)(void* self, void* names);
LANTERN_API void* (LANTERN_PTR lantern_rename_tensor_dimnamelist)(void* self, void* names);
LANTERN_API void* (LANTERN_PTR lantern_align_to_tensor_dimnamelist)(void* self, void* names);
LANTERN_API void* (LANTERN_PTR lantern_align_as_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_align_tensors_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_refine_names_tensor_dimnamelist)(void* self, void* names);
LANTERN_API void* (LANTERN_PTR lantern_unflatten_tensor_dimname_intarrayref_dimnamelist)(void* self, void* dim, void* sizes, void* names);
LANTERN_API void* (LANTERN_PTR lantern_unflatten_tensor_intt_intarrayref_dimnamelist)(void* self, void* dim, void* sizes, void* names);
LANTERN_API void* (LANTERN_PTR lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* deterministic, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_bool_bool)(void* weight_arr, void* weight_stride0, void* input_size, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor)(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state);
LANTERN_API void* (LANTERN_PTR lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool)(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions)(void* dropout, void* train, void* dropout_seed, void* options);
LANTERN_API void* (LANTERN_PTR lantern__debug_has_internal_overlap_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__fused_dropout_tensor_double_generator)(void* self, void* p, void* generator);
LANTERN_API void* (LANTERN_PTR lantern__masked_scale_tensor_tensor_double)(void* self, void* mask, void* scale);
LANTERN_API void* (LANTERN_PTR lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype)(void* quasi, void* n, void* sobolstate, void* dimension, void* num_generated, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt)(void* self, void* n, void* sobolstate, void* dimension, void* num_generated);
LANTERN_API void* (LANTERN_PTR lantern__sobol_engine_scramble__tensor_tensor_intt)(void* self, void* ltm, void* dimension);
LANTERN_API void* (LANTERN_PTR lantern__sobol_engine_initialize_state__tensor_intt)(void* self, void* dimension);
LANTERN_API void* (LANTERN_PTR lantern__reshape_from_tensor_tensor_tensor)(void* self, void* shape);
LANTERN_API void* (LANTERN_PTR lantern__shape_as_tensor_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_dropout_tensor_double_bool)(void* input, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_dropout__tensor_double_bool)(void* self, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_feature_dropout_tensor_double_bool)(void* input, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_feature_dropout__tensor_double_bool)(void* self, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_alpha_dropout_tensor_double_bool)(void* input, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_alpha_dropout__tensor_double_bool)(void* self, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_feature_alpha_dropout_tensor_double_bool)(void* input, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_feature_alpha_dropout__tensor_double_bool)(void* self, void* p, void* train);
LANTERN_API void* (LANTERN_PTR lantern_abs_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_abs__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_abs_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_acos_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_acos__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_acos_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool)(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool1d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool1d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_add_tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_add__tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_add_out_tensor_tensor_tensor_scalar)(void* out, void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_add_tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_add__tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmv_tensor_tensor_tensor_scalar_scalar)(void* self, void* mat, void* vec, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmv__tensor_tensor_tensor_scalar_scalar)(void* self, void* mat, void* vec, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* mat, void* vec, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addr_tensor_tensor_tensor_scalar_scalar)(void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addr__tensor_tensor_tensor_scalar_scalar)(void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_affine_grid_generator_tensor_intarrayref_bool)(void* theta, void* size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_affine_grid_generator_backward_tensor_intarrayref_bool)(void* grad, void* size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_all_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_all_out_tensor_tensor_intt_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_all_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_all_out_tensor_tensor_dimname_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_allclose_tensor_tensor_double_double_bool)(void* self, void* other, void* rtol, void* atol, void* equal_nan);
LANTERN_API void* (LANTERN_PTR lantern_any_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_any_out_tensor_tensor_intt_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_any_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_any_out_tensor_tensor_dimname_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_tensoroptions)(void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_scalar_tensoroptions)(void* start, void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_scalar_scalar_tensoroptions)(void* start, void* end, void* step, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_out_tensor_scalar)(void* out, void* end);
LANTERN_API void* (LANTERN_PTR lantern_arange_out_tensor_scalar_scalar_scalar)(void* out, void* start, void* end, void* step);
LANTERN_API void* (LANTERN_PTR lantern__dim_arange_tensor_intt)(void* like, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_argmax_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_argmin_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_as_strided_tensor_intarrayref_intarrayref_intt)(void* self, void* size, void* stride, void* storage_offset);
LANTERN_API void* (LANTERN_PTR lantern_as_strided__tensor_intarrayref_intarrayref_intt)(void* self, void* size, void* stride, void* storage_offset);
LANTERN_API void* (LANTERN_PTR lantern_asin_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_asin__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_asin_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_atan_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_atan__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_atan_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_baddbmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_baddbmm__tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_bartlett_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_bartlett_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool)(void* impl_index, void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var_transform, void* train, void* eps, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli_tensor_generator)(void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli_out_tensor_tensor_generator)(void* out, void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli__tensor_tensor_generator)(void* self, void* p, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli__tensor_double_generator)(void* self, void* p, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli_tensor_double_generator)(void* self, void* p, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bilinear_tensor_tensor_tensor_tensor)(void* input1, void* input2, void* weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt)(void* self, void* target, void* weight, void* pos_weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_bincount_tensor_tensor_intt)(void* self, void* weights, void* minlength);
LANTERN_API void* (LANTERN_PTR lantern_bitwise_not_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_bitwise_not__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_bitwise_not_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_logical_not_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_logical_not__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_logical_not_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_logical_xor_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_logical_xor__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_logical_xor_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_blackman_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_blackman_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_bmm_tensor_tensor)(void* self, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_bmm_out_tensor_tensor_tensor)(void* out, void* self, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_broadcast_tensors_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_cat_tensorlist_intt)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_out_tensor_tensorlist_intt)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_tensorlist_dimname)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_out_tensor_tensorlist_dimname)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_ceil_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_ceil__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_ceil_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_chain_matmul_tensorlist)(void* matrices);
LANTERN_API void* (LANTERN_PTR lantern_chunk_tensor_intt_intt)(void* self, void* chunks, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_clamp_tensor_scalar_scalar)(void* self, void* min, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp__tensor_scalar_scalar)(void* self, void* min, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp_out_tensor_tensor_scalar_scalar)(void* out, void* self, void* min, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp_max_tensor_scalar)(void* self, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp_max__tensor_scalar)(void* self, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp_max_out_tensor_tensor_scalar)(void* out, void* self, void* max);
LANTERN_API void* (LANTERN_PTR lantern_clamp_min_tensor_scalar)(void* self, void* min);
LANTERN_API void* (LANTERN_PTR lantern_clamp_min__tensor_scalar)(void* self, void* min);
LANTERN_API void* (LANTERN_PTR lantern_clamp_min_out_tensor_tensor_scalar)(void* out, void* self, void* min);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_is_acceptable_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_constant_pad_nd_tensor_intarrayref_scalar)(void* self, void* pad, void* value);
LANTERN_API void* (LANTERN_PTR lantern_contiguous_tensor_memoryformat)(void* self, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool)(void* grad_output, void* input, void* weight, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding);
LANTERN_API void* (LANTERN_PTR lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_stdarraybool)(void* ggI, void* ggW, void* ggb, void* gO, void* weight, void* self, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv_tbc_tensor_tensor_tensor_intt)(void* self, void* weight, void* bias, void* pad);
LANTERN_API void* (LANTERN_PTR lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt)(void* self, void* input, void* weight, void* bias, void* pad);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_copy__tensor_tensor_bool)(void* self, void* src, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern__copy_from_tensor_tensor_bool)(void* self, void* dst, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern_cos_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_cos__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_cos_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_cosh_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_cosh__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_cosh_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt)(void* input1, void* input2, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt)(void* theta, void* N, void* C, void* H, void* W);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt)(void* grad, void* N, void* C, void* H, void* W);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double)(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_backward_bias_tensor)(void* grad_output);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_bias_tensor)(void* grad_output);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_grid_sampler_tensor_tensor)(void* self, void* grid);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor)(void* self, void* grid, void* grad_output);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_tensor_intt_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_out_tensor_tensor_intt_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_tensor_dimname_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_out_tensor_tensor_dimname_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_tensor_intt_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_out_tensor_tensor_intt_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_tensor_dimname_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_out_tensor_tensor_dimname_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool)(void* grad, void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* neg_log_likelihood, void* log_alpha, void* blank, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern_det_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_diag_embed_tensor_intt_intt_intt)(void* self, void* offset, void* dim1, void* dim2);
LANTERN_API void* (LANTERN_PTR lantern_diagflat_tensor_intt)(void* self, void* offset);
LANTERN_API void* (LANTERN_PTR lantern_diagonal_tensor_intt_intt_intt)(void* self, void* offset, void* dim1, void* dim2);
LANTERN_API void* (LANTERN_PTR lantern_fill_diagonal__tensor_scalar_bool)(void* self, void* fill_value, void* wrap);
LANTERN_API void* (LANTERN_PTR lantern_div_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_div__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_div_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_div_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_div__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_dot_tensor_tensor)(void* self, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_dot_out_tensor_tensor_tensor)(void* out, void* self, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_einsum_stdstring_tensorlist)(void* equation, void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_embedding_tensor_tensor_intt_bool_bool)(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse);
LANTERN_API void* (LANTERN_PTR lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool)(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse);
LANTERN_API void* (LANTERN_PTR lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool)(void* grad_output, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq);
LANTERN_API void* (LANTERN_PTR lantern_embedding_renorm__tensor_tensor_double_double)(void* self, void* indices, void* max_norm, void* norm_type);
LANTERN_API void* (LANTERN_PTR lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool)(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq);
LANTERN_API void* (LANTERN_PTR lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor)(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor)(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor)(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt)(void* grad, void* weight, void* indices, void* offsets, void* offset2bag, void* mode);
LANTERN_API void* (LANTERN_PTR lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat)(void* size, void* names, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_empty_intarrayref_tensoroptions_memoryformat)(void* size, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_new_empty_tensor_intarrayref_tensoroptions)(void* self, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_new_full_tensor_intarrayref_scalar_tensoroptions)(void* self, void* size, void* fill_value, void* options);
LANTERN_API void* (LANTERN_PTR lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat)(void* size, void* options, void* scale, void* zero_point, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat)(void* size, void* scales, void* zero_points, void* axis, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_resize__tensor_intarrayref)(void* self, void* size);
LANTERN_API void* (LANTERN_PTR lantern_empty_out_tensor_intarrayref_memoryformat)(void* out, void* size, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_empty_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_empty_like_tensor_tensoroptions_memoryformat)(void* self, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_empty_strided_intarrayref_intarrayref_tensoroptions)(void* size, void* stride, void* options);
LANTERN_API void* (LANTERN_PTR lantern_erf_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erf__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erf_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfc_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfc__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfc_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_exp_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_exp__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_exp_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_expm1_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_expm1__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_expm1_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_expand_tensor_intarrayref_bool)(void* self, void* size, void* implicit);
LANTERN_API void* (LANTERN_PTR lantern_expand_as_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eye_intt_tensoroptions)(void* n, void* options);
LANTERN_API void* (LANTERN_PTR lantern_eye_intt_intt_tensoroptions)(void* n, void* m, void* options);
LANTERN_API void* (LANTERN_PTR lantern_eye_out_tensor_intt)(void* out, void* n);
LANTERN_API void* (LANTERN_PTR lantern_eye_out_tensor_intt_intt)(void* out, void* n, void* m);
LANTERN_API void* (LANTERN_PTR lantern_flatten_tensor_intt_intt)(void* self, void* start_dim, void* end_dim);
LANTERN_API void* (LANTERN_PTR lantern_flatten_tensor_intt_intt_dimname)(void* self, void* start_dim, void* end_dim, void* out_dim);
LANTERN_API void* (LANTERN_PTR lantern_flatten_tensor_dimname_dimname_dimname)(void* self, void* start_dim, void* end_dim, void* out_dim);
LANTERN_API void* (LANTERN_PTR lantern_flatten_tensor_dimnamelist_dimname)(void* self, void* dims, void* out_dim);
LANTERN_API void* (LANTERN_PTR lantern_fill__tensor_scalar)(void* self, void* value);
LANTERN_API void* (LANTERN_PTR lantern_fill__tensor_tensor)(void* self, void* value);
LANTERN_API void* (LANTERN_PTR lantern_floor_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_floor__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_floor_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_frac_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_frac__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_frac_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_full_intarrayref_scalar_dimnamelist_tensoroptions)(void* size, void* fill_value, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_full_intarrayref_scalar_tensoroptions)(void* size, void* fill_value, void* options);
LANTERN_API void* (LANTERN_PTR lantern_full_out_tensor_intarrayref_scalar)(void* out, void* size, void* fill_value);
LANTERN_API void* (LANTERN_PTR lantern_full_like_tensor_scalar)(void* self, void* fill_value);
LANTERN_API void* (LANTERN_PTR lantern_full_like_tensor_scalar_tensoroptions)(void* self, void* fill_value, void* options);
LANTERN_API void* (LANTERN_PTR lantern_from_file_stdstring_bool_intt_tensoroptions)(void* filename, void* shared, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_tensor_tensor_intt_intt_bool)(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool)(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool)(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool)(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool)(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_hann_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hann_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_double_tensoroptions)(void* window_length, void* periodic, void* alpha, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_double_double_tensoroptions)(void* window_length, void* periodic, void* alpha, void* beta, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hinge_embedding_loss_tensor_tensor_double_intt)(void* self, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_ger_tensor_tensor)(void* self, void* vec2);
LANTERN_API void* (LANTERN_PTR lantern_ger_out_tensor_tensor_tensor)(void* out, void* self, void* vec2);
LANTERN_API void* (LANTERN_PTR lantern_group_norm_tensor_intt_tensor_tensor_double_bool)(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern_fft_tensor_intt_bool)(void* self, void* signal_ndim, void* normalized);
LANTERN_API void* (LANTERN_PTR lantern_ifft_tensor_intt_bool)(void* self, void* signal_ndim, void* normalized);
LANTERN_API void* (LANTERN_PTR lantern_rfft_tensor_intt_bool_bool)(void* self, void* signal_ndim, void* normalized, void* onesided);
LANTERN_API void* (LANTERN_PTR lantern_irfft_tensor_intt_bool_bool_intarrayref)(void* self, void* signal_ndim, void* normalized, void* onesided, void* signal_sizes);
LANTERN_API void* (LANTERN_PTR lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_bool_bool_intarrayref)(void* self, void* signal_ndim, void* complex_input, void* complex_output, void* inverse, void* checked_signal_sizes, void* normalized, void* onesided, void* output_sizes);
LANTERN_API void* (LANTERN_PTR lantern__cufft_get_plan_cache_size_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern__cufft_get_plan_cache_max_size_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern__cufft_set_plan_cache_max_size_intt_intt)(void* device_index, void* max_size);
LANTERN_API void* (LANTERN_PTR lantern__cufft_clear_plan_cache_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern_index_tensor_tensorlist)(void* self, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_index_copy__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_copy_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_copy__tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_copy_tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_put__tensor_tensorlist_tensor_bool)(void* self, void* indices, void* values, void* accumulate);
LANTERN_API void* (LANTERN_PTR lantern_index_put_tensor_tensorlist_tensor_bool)(void* self, void* indices, void* values, void* accumulate);
LANTERN_API void* (LANTERN_PTR lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool)(void* self, void* indices, void* values, void* accumulate, void* unsafe);
LANTERN_API void* (LANTERN_PTR lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern_inverse_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_inverse_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern__inverse_helper_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_isclose_tensor_tensor_double_double_bool)(void* self, void* other, void* rtol, void* atol, void* equal_nan);
LANTERN_API void* (LANTERN_PTR lantern_isnan_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_distributed_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_floating_point_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_complex_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_nonzero_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_same_size_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_is_signed_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_kl_div_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_kl_div_backward_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_kthvalue_tensor_intt_intt_bool)(void* self, void* k, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool)(void* values, void* indices, void* self, void* k, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_kthvalue_tensor_intt_dimname_bool)(void* self, void* k, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool)(void* values, void* indices, void* self, void* k, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool)(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable);
LANTERN_API void* (LANTERN_PTR lantern_native_layer_norm_tensor_tensor_tensor_intt_intt_double)(void* input, void* weight, void* bias, void* M, void* N, void* eps);
LANTERN_API void* (LANTERN_PTR lantern_native_layer_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool)(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* M, void* N, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_native_layer_norm_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool)(void* ggI, void* ggW, void* ggb, void* gO, void* input, void* mean, void* rstd, void* weight, void* M, void* N, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_linear_tensor_tensor_tensor)(void* input, void* weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_linear_tensor_tensor_tensor)(void* input, void* weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor)(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor)(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_quantize_weight_tensor)(void* input);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_gemm_matrix_fp16_tensor)(void* input);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor)(void* input, void* packed_weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor)(void* input, void* packed_weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix_tensor)(void* input);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt)(void* input, void* K, void* N);
LANTERN_API void* (LANTERN_PTR lantern_linspace_scalar_scalar_intt_tensoroptions)(void* start, void* end, void* steps, void* options);
LANTERN_API void* (LANTERN_PTR lantern_linspace_out_tensor_scalar_scalar_intt)(void* out, void* start, void* end, void* steps);
LANTERN_API void* (LANTERN_PTR lantern_log_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log10_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log10__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log10_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log1p_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log1p__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log1p_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log2_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log2__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log2_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_logdet_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_logspace_scalar_scalar_intt_double_tensoroptions)(void* start, void* end, void* steps, void* base, void* options);
LANTERN_API void* (LANTERN_PTR lantern_logspace_out_tensor_scalar_scalar_intt_double)(void* out, void* start, void* end, void* steps, void* base);
LANTERN_API void* (LANTERN_PTR lantern_log_softmax_tensor_intt_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_log_softmax_tensor_dimname_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__log_softmax_tensor_intt_bool)(void* self, void* dim, void* half_to_float);
LANTERN_API void* (LANTERN_PTR lantern__log_softmax_backward_data_tensor_tensor_intt_tensor)(void* grad_output, void* output, void* dim, void* self);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_tensor_dimnamelist_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_out_tensor_tensor_dimnamelist_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt)(void* input1, void* input2, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_matmul_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_matmul_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_matrix_rank_tensor_double_bool)(void* self, void* tol, void* symmetric);
LANTERN_API void* (LANTERN_PTR lantern_matrix_rank_tensor_bool)(void* self, void* symmetric);
LANTERN_API void* (LANTERN_PTR lantern_matrix_power_tensor_intt)(void* self, void* n);
LANTERN_API void* (LANTERN_PTR lantern_max_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_out_tensor_tensor_tensor_intt_bool)(void* max, void* max_values, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_values_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_out_tensor_tensor_tensor_dimname_bool)(void* max, void* max_values, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_values_tensor_dimnamelist_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_mean_tensor_scalartype)(void* self, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_mean_tensor_intarrayref_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_mean_tensor_dimnamelist_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_median_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_median_out_tensor_tensor_tensor_intt_bool)(void* values, void* indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_median_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_median_out_tensor_tensor_tensor_dimname_bool)(void* values, void* indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_out_tensor_tensor_tensor_intt_bool)(void* min, void* min_indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_values_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_out_tensor_tensor_tensor_dimname_bool)(void* min, void* min_indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_min_values_tensor_dimnamelist_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* bias_defined);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon);
LANTERN_API void* (LANTERN_PTR lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double)(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_backward_bias_tensor)(void* grad_output);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic);
LANTERN_API void* (LANTERN_PTR lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor)(void* input, void* weight, void* weight_stride0, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state);
LANTERN_API void* (LANTERN_PTR lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool)(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_mm_tensor_tensor)(void* self, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_mm_out_tensor_tensor_tensor)(void* out, void* self, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern__sparse_mm_tensor_tensor)(void* sparse, void* dense);
LANTERN_API void* (LANTERN_PTR lantern_mode_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_mode_out_tensor_tensor_tensor_intt_bool)(void* values, void* indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_mode_tensor_dimname_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_mode_out_tensor_tensor_tensor_dimname_bool)(void* values, void* indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_mul_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_mul__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_mul_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_mul_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_mul__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_mv_tensor_tensor)(void* self, void* vec);
LANTERN_API void* (LANTERN_PTR lantern_mv_out_tensor_tensor_tensor)(void* out, void* self, void* vec);
LANTERN_API void* (LANTERN_PTR lantern_mvlgamma_tensor_intt)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern_mvlgamma__tensor_intt)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern_narrow_copy_tensor_intt_intt_intt)(void* self, void* dim, void* start, void* length);
LANTERN_API void* (LANTERN_PTR lantern_narrow_tensor_intt_intt_intt)(void* self, void* dim, void* start, void* length);
LANTERN_API void* (LANTERN_PTR lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_stats_tensor_double)(void* input, void* eps);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double)(void* input, void* weight, void* bias, void* mean, void* invstd, void* eps);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt)(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* count);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_intarrayref)(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* counts);
LANTERN_API void* (LANTERN_PTR lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool)(void* grad_out, void* input, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_invstd, void* train, void* eps, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool)(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* input_g, void* weight_g, void* bias_g);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor)(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* mean_dy, void* mean_dy_xmu);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_update_stats_tensor_tensor_tensor_double)(void* input, void* running_mean, void* running_var, void* momentum);
LANTERN_API void* (LANTERN_PTR lantern__nnpack_available)();
LANTERN_API void* (LANTERN_PTR lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref)(void* input, void* weight, void* bias, void* padding);
LANTERN_API void* (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool)(void* input, void* grad_output, void* weight, void* padding, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref)(void* input, void* grad_output, void* weight, void* padding);
LANTERN_API void* (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref)(void* input, void* weightsize, void* grad_output, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_ones_intarrayref_dimnamelist_tensoroptions)(void* size, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_ones_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_ones_out_tensor_intarrayref)(void* out, void* size);
LANTERN_API void* (LANTERN_PTR lantern_ones_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_ones_like_tensor_tensoroptions)(void* self, void* options);
LANTERN_API void* (LANTERN_PTR lantern_pairwise_distance_tensor_tensor_double_double_bool)(void* x1, void* x2, void* p, void* eps, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_cdist_tensor_tensor_double)(void* x1, void* x2, void* p);
LANTERN_API void* (LANTERN_PTR lantern__cdist_backward_tensor_tensor_tensor_double_tensor)(void* grad, void* x1, void* x2, void* p, void* cdist);
LANTERN_API void* (LANTERN_PTR lantern_pdist_tensor_double)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern__pdist_forward_tensor_double)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern__pdist_backward_tensor_tensor_double_tensor)(void* grad, void* self, void* p, void* pdist);
LANTERN_API void* (LANTERN_PTR lantern_cosine_similarity_tensor_tensor_intt_double)(void* x1, void* x2, void* dim, void* eps);
LANTERN_API void* (LANTERN_PTR lantern_permute_tensor_intarrayref)(void* self, void* dims);
LANTERN_API void* (LANTERN_PTR lantern_numpy_t_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_pixel_shuffle_tensor_intt)(void* self, void* upscale_factor);
LANTERN_API void* (LANTERN_PTR lantern_is_pinned_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_pin_memory_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_pinverse_tensor_double)(void* self, void* rcond);
LANTERN_API void* (LANTERN_PTR lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt)(void* input, void* target, void* log_input, void* full, void* eps, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_scalar_tensor_scalar_tensoroptions)(void* s, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rand_intarrayref_dimnamelist_tensoroptions)(void* size, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rand_intarrayref_generator_dimnamelist_tensoroptions)(void* size, void* generator, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rand_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rand_intarrayref_generator_tensoroptions)(void* size, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rand_out_tensor_intarrayref)(void* out, void* size);
LANTERN_API void* (LANTERN_PTR lantern_rand_out_tensor_intarrayref_generator)(void* out, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_rand_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_rand_like_tensor_tensoroptions)(void* self, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_intt_intarrayref_tensoroptions)(void* high, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_intt_intarrayref_generator_tensoroptions)(void* high, void* size, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_intt_intt_intarrayref_tensoroptions)(void* low, void* high, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_intt_intt_intarrayref_generator_tensoroptions)(void* low, void* high, void* size, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_out_tensor_intt_intarrayref)(void* out, void* high, void* size);
LANTERN_API void* (LANTERN_PTR lantern_randint_out_tensor_intt_intarrayref_generator)(void* out, void* high, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_randint_out_tensor_intt_intt_intarrayref)(void* out, void* low, void* high, void* size);
LANTERN_API void* (LANTERN_PTR lantern_randint_out_tensor_intt_intt_intarrayref_generator)(void* out, void* low, void* high, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_randint_like_tensor_intt)(void* self, void* high);
LANTERN_API void* (LANTERN_PTR lantern_randint_like_tensor_intt_intt)(void* self, void* low, void* high);
LANTERN_API void* (LANTERN_PTR lantern_randint_like_tensor_intt_tensoroptions)(void* self, void* high, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randint_like_tensor_intt_intt_tensoroptions)(void* self, void* low, void* high, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randn_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randn_intarrayref_generator_tensoroptions)(void* size, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randn_intarrayref_dimnamelist_tensoroptions)(void* size, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randn_intarrayref_generator_dimnamelist_tensoroptions)(void* size, void* generator, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randn_out_tensor_intarrayref)(void* out, void* size);
LANTERN_API void* (LANTERN_PTR lantern_randn_out_tensor_intarrayref_generator)(void* out, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_randn_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_randn_like_tensor_tensoroptions)(void* self, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randperm_intt_tensoroptions)(void* n, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randperm_intt_generator_tensoroptions)(void* n, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_randperm_out_tensor_intt)(void* out, void* n);
LANTERN_API void* (LANTERN_PTR lantern_randperm_out_tensor_intt_generator)(void* out, void* n, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_range_scalar_scalar_scalar_tensoroptions)(void* start, void* end, void* step, void* options);
LANTERN_API void* (LANTERN_PTR lantern_range_scalar_scalar_tensoroptions)(void* start, void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_range_out_tensor_scalar_scalar_scalar)(void* out, void* start, void* end, void* step);
LANTERN_API void* (LANTERN_PTR lantern_reciprocal_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_reciprocal__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_reciprocal_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_neg_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_neg__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_neg_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_repeat_tensor_intarrayref)(void* self, void* repeats);
LANTERN_API void* (LANTERN_PTR lantern_repeat_interleave_tensor)(void* repeats);
LANTERN_API void* (LANTERN_PTR lantern_repeat_interleave_tensor_tensor_intt)(void* self, void* repeats, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_repeat_interleave_tensor_intt_intt)(void* self, void* repeats, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_reshape_tensor_intarrayref)(void* self, void* shape);
LANTERN_API void* (LANTERN_PTR lantern__mkldnn_reshape_tensor_intarrayref)(void* self, void* shape);
LANTERN_API void* (LANTERN_PTR lantern_reshape_as_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_round_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_round__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_round_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_tensor_scalar_scalar_bool_generator)(void* self, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_rrelu__tensor_scalar_scalar_bool_generator)(void* self, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_relu_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_relu__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_prelu_tensor_tensor)(void* self, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_prelu_backward_tensor_tensor_tensor)(void* grad_output, void* self, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_gelu_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_gelu_backward_tensor_tensor)(void* grad, void* self);
LANTERN_API void* (LANTERN_PTR lantern_hardshrink_tensor_scalar)(void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_hardshrink_backward_tensor_tensor_scalar)(void* grad_out, void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_rsqrt_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_rsqrt__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_rsqrt_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_select_tensor_dimname_intt)(void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_select_tensor_intt_intt)(void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_selu_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_selu__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_celu_tensor_scalar)(void* self, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_celu__tensor_scalar)(void* self, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sigmoid_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sigmoid__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sigmoid_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_sin_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sin__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sin_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_sinh_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sinh__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sinh_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_detach_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_detach__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_size_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_size_tensor_dimname)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_slice_tensor_intt_intt_intt_intt)(void* self, void* dim, void* start, void* end, void* step);
LANTERN_API void* (LANTERN_PTR lantern_slogdet_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_smm_tensor_tensor)(void* self, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_softmax_tensor_intt_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_softmax_tensor_dimname_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__softmax_tensor_intt_bool)(void* self, void* dim, void* half_to_float);
LANTERN_API void* (LANTERN_PTR lantern__softmax_backward_data_tensor_tensor_intt_tensor)(void* grad_output, void* output, void* dim, void* self);
LANTERN_API void* (LANTERN_PTR lantern_split_tensor_intt_intt)(void* self, void* split_size, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_split_with_sizes_tensor_intarrayref_intt)(void* self, void* split_sizes, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_squeeze_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_squeeze_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_squeeze_tensor_dimname)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_squeeze__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_squeeze__tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_squeeze__tensor_dimname)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* mat1, void* mat2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_stack_tensorlist_intt)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_stack_out_tensor_tensorlist_intt)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_stft_tensor_intt_intt_intt_tensor_bool_bool)(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided);
LANTERN_API void* (LANTERN_PTR lantern_stride_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_stride_tensor_dimname)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_sum_tensor_scalartype)(void* self, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_tensor_intarrayref_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_tensor_dimnamelist_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_to_size_tensor_intarrayref)(void* self, void* size);
LANTERN_API void* (LANTERN_PTR lantern_sqrt_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sqrt__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sqrt_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_std_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern_std_tensor_intarrayref_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_mean_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern_std_mean_tensor_intarrayref_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_mean_tensor_dimnamelist_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_out_tensor_tensor_intarrayref_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_tensor_dimnamelist_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_out_tensor_tensor_dimnamelist_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_prod_tensor_scalartype)(void* self, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_prod_tensor_intt_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_prod_out_tensor_tensor_intt_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_prod_tensor_dimname_bool_scalartype)(void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_prod_out_tensor_tensor_dimname_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_t_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_t__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_tan_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_tan__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_tan_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_tanh_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_tanh__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_tanh_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_tensordot_tensor_tensor_intarrayref_intarrayref)(void* self, void* other, void* dims_self, void* dims_other);
LANTERN_API void* (LANTERN_PTR lantern_threshold_tensor_scalar_scalar)(void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold__tensor_scalar_scalar)(void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold_out_tensor_tensor_scalar_scalar)(void* out, void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold_backward_tensor_tensor_scalar)(void* grad_output, void* self, void* threshold);
LANTERN_API void* (LANTERN_PTR lantern_transpose_tensor_intt_intt)(void* self, void* dim0, void* dim1);
LANTERN_API void* (LANTERN_PTR lantern_transpose_tensor_dimname_dimname)(void* self, void* dim0, void* dim1);
LANTERN_API void* (LANTERN_PTR lantern__mkldnn_transpose_tensor_intt_intt)(void* self, void* dim0, void* dim1);
LANTERN_API void* (LANTERN_PTR lantern_transpose__tensor_intt_intt)(void* self, void* dim0, void* dim1);
LANTERN_API void* (LANTERN_PTR lantern__mkldnn_transpose__tensor_intt_intt)(void* self, void* dim0, void* dim1);
LANTERN_API void* (LANTERN_PTR lantern_one_hot_tensor_intt)(void* self, void* num_classes);
LANTERN_API void* (LANTERN_PTR lantern_flip_tensor_intarrayref)(void* self, void* dims);
LANTERN_API void* (LANTERN_PTR lantern_roll_tensor_intarrayref_intarrayref)(void* self, void* shifts, void* dims);
LANTERN_API void* (LANTERN_PTR lantern_rot90_tensor_intt_intarrayref)(void* self, void* k, void* dims);
LANTERN_API void* (LANTERN_PTR lantern_trapz_tensor_tensor_intt)(void* y, void* x, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_trapz_tensor_double_intt)(void* y, void* dx, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt)(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim);
LANTERN_API void* (LANTERN_PTR lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt)(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_trunc_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_trunc__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_trunc_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_type_as_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern__has_compatible_shallow_copy_type_tensor_tensor)(void* self, void* from);
LANTERN_API void* (LANTERN_PTR lantern__unique_tensor_bool_bool)(void* self, void* sorted, void* return_inverse);
LANTERN_API void* (LANTERN_PTR lantern_unique_dim_tensor_intt_bool_bool_bool)(void* self, void* dim, void* sorted, void* return_inverse, void* return_counts);
LANTERN_API void* (LANTERN_PTR lantern_unique_consecutive_tensor_bool_bool_intt)(void* self, void* return_inverse, void* return_counts, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_unique_dim_consecutive_tensor_intt_bool_bool)(void* self, void* dim, void* return_inverse, void* return_counts);
LANTERN_API void* (LANTERN_PTR lantern__unique2_tensor_bool_bool_bool)(void* self, void* sorted, void* return_inverse, void* return_counts);
LANTERN_API void* (LANTERN_PTR lantern__unsafe_view_tensor_intarrayref)(void* self, void* size);
LANTERN_API void* (LANTERN_PTR lantern_unsqueeze_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_unsqueeze__tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_var_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern_var_tensor_intarrayref_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_out_tensor_tensor_intarrayref_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_tensor_dimnamelist_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_out_tensor_tensor_dimnamelist_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_mean_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern_var_mean_tensor_intarrayref_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_mean_tensor_dimnamelist_bool_bool)(void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_view_as_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_where_tensor_tensor_tensor)(void* condition, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_where_tensor)(void* condition);
LANTERN_API void* (LANTERN_PTR lantern__s_where_tensor_tensor_tensor)(void* condition, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_norm_except_dim_tensor_intt_intt)(void* v, void* pow, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__weight_norm_tensor_tensor_intt)(void* v, void* g, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__weight_norm_cuda_interface_tensor_tensor_intt)(void* v, void* g, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt)(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt)(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_zeros_intarrayref_dimnamelist_tensoroptions)(void* size, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_zeros_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_zeros_out_tensor_intarrayref)(void* out, void* size);
LANTERN_API void* (LANTERN_PTR lantern_zeros_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_zeros_like_tensor_tensoroptions)(void* self, void* options);
LANTERN_API void* (LANTERN_PTR lantern__standard_gamma_grad_tensor_tensor)(void* self, void* output);
LANTERN_API void* (LANTERN_PTR lantern__standard_gamma_tensor_generator)(void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern__dirichlet_grad_tensor_tensor_tensor)(void* x, void* alpha, void* total);
LANTERN_API void* (LANTERN_PTR lantern__sample_dirichlet_tensor_generator)(void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_poisson_tensor_generator)(void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_native_norm_tensor_scalar)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_scalartype)(void* self, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_intarrayref)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_intarrayref_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_backward_tensor_tensor_intarrayref)(void* grad, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar_scalartype)(void* self, void* p, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar)(void* self, void* p);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar_intarrayref_bool_scalartype)(void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar_intarrayref_bool)(void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype)(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_intarrayref_bool)(void* out, void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar_dimnamelist_bool_scalartype)(void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_tensor_scalar_dimnamelist_bool)(void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype)(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool)(void* out, void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_tensor_bool)(void* self, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_out_tensor_tensor_bool)(void* out, void* self, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_clone_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_resize_as__tensor_tensor)(void* self, void* the_template);
LANTERN_API void* (LANTERN_PTR lantern_pow_out_tensor_tensor_scalar)(void* out, void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_pow_tensor_scalar)(void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_zero__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sub_out_tensor_tensor_tensor_scalar)(void* out, void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sub_tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sub__tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sub_tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sub__tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_rsub_tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_rsub_tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* sparse, void* dense, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* mat1, void* mat2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addmm__tensor_tensor_tensor_scalar_scalar)(void* self, void* mat1, void* mat2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_tensor_tensor_tensoroptions)(void* indices, void* values, void* options);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions)(void* indices, void* values, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions)(void* indices, void* values, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions)(void* sparse_dim, void* dense_dim, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions)(void* sparse_dim, void* dense_dim, void* size, void* indices, void* values, void* options);
LANTERN_API void* (LANTERN_PTR lantern_sparse_resize__tensor_intarrayref_intt_intt)(void* self, void* size, void* sparse_dim, void* dense_dim);
LANTERN_API void* (LANTERN_PTR lantern_sparse_resize_and_clear__tensor_intarrayref_intt_intt)(void* self, void* size, void* sparse_dim, void* dense_dim);
LANTERN_API void* (LANTERN_PTR lantern_sparse_mask_tensor_tensor)(void* self, void* mask);
LANTERN_API void* (LANTERN_PTR lantern_to_dense_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_to_dense_backward_tensor_tensor)(void* grad, void* input);
LANTERN_API void* (LANTERN_PTR lantern_sparse_dim_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__dimi_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_dense_dim_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__dimv_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__nnz_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_coalesce_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_is_coalesced_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__indices_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__values_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__coalesced__tensor_bool)(void* self, void* coalesced);
LANTERN_API void* (LANTERN_PTR lantern_indices_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_values_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_hspmm_out_tensor_tensor_tensor)(void* out, void* mat1, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_hspmm_tensor_tensor)(void* mat1, void* mat2);
LANTERN_API void* (LANTERN_PTR lantern_copy_sparse_to_sparse__tensor_tensor_bool)(void* self, void* src, void* non_blocking);
LANTERN_API void* (LANTERN_PTR lantern_numel_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_unbind_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_unbind_tensor_dimname)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_to_sparse_tensor_intt)(void* self, void* sparse_dim);
LANTERN_API void* (LANTERN_PTR lantern_to_sparse_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_to_mkldnn_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt)(void* self, void* padding, void* stride, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_to_mkldnn_backward_tensor_tensor)(void* grad, void* input);
LANTERN_API void* (LANTERN_PTR lantern_quantize_per_tensor_tensor_double_intt_scalartype)(void* self, void* scale, void* zero_point, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype)(void* self, void* scales, void* zero_points, void* axis, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_dequantize_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_q_scale_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_q_zero_point_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_q_per_channel_scales_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_q_per_channel_zero_points_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_q_per_channel_axis_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_int_repr_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__make_per_tensor_quantized_tensor_tensor_double_intt)(void* self, void* scale, void* zero_point);
LANTERN_API void* (LANTERN_PTR lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt)(void* self, void* scale, void* zero_point, void* axis);
LANTERN_API void* (LANTERN_PTR lantern_qscheme_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt)(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max);
LANTERN_API void* (LANTERN_PTR lantern_fake_quantize_per_tensor_affine_backward_tensor_tensor_double_intt_intt_intt)(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max);
LANTERN_API void* (LANTERN_PTR lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt)(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max);
LANTERN_API void* (LANTERN_PTR lantern_fake_quantize_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt)(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max);
LANTERN_API void* (LANTERN_PTR lantern_to_tensor_tensoroptions_bool_bool)(void* self, void* options, void* non_blocking, void* copy);
LANTERN_API void* (LANTERN_PTR lantern_to_tensor_device_scalartype_bool_bool)(void* self, void* device, void* dtype, void* non_blocking, void* copy);
LANTERN_API void* (LANTERN_PTR lantern_to_tensor_scalartype_bool_bool)(void* self, void* dtype, void* non_blocking, void* copy);
LANTERN_API void* (LANTERN_PTR lantern_to_tensor_tensor_bool_bool)(void* self, void* other, void* non_blocking, void* copy);
LANTERN_API void* (LANTERN_PTR lantern_meshgrid_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_cartesian_prod_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_combinations_tensor_intt_bool)(void* self, void* r, void* with_replacement);
LANTERN_API void* (LANTERN_PTR lantern_item_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_result_type_tensor_tensor)(void* tensor, void* other);
LANTERN_API void* (LANTERN_PTR lantern_result_type_tensor_scalar)(void* tensor, void* other);
LANTERN_API void* (LANTERN_PTR lantern_result_type_scalar_tensor)(void* scalar, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_result_type_scalar_scalar)(void* scalar1, void* scalar2);
LANTERN_API void* (LANTERN_PTR lantern_can_cast_scalartype_scalartype)(void* from, void* to);
LANTERN_API void* (LANTERN_PTR lantern_promote_types_scalartype_scalartype)(void* type1, void* type2);
LANTERN_API void* (LANTERN_PTR lantern__local_scalar_dense_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor)(void* input_gates, void* hidden_gates, void* cx, void* input_bias, void* hidden_bias);
LANTERN_API void* (LANTERN_PTR lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool)(void* grad_hy, void* grad_cy, void* cx, void* cy, void* workspace, void* has_bias);
LANTERN_API void* (LANTERN_PTR lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor)(void* grad_hy, void* grad_cy, void* input_gates, void* hidden_gates, void* input_bias, void* hidden_bias, void* cx, void* cy);
LANTERN_API void* (LANTERN_PTR lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor)(void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias);
LANTERN_API void* (LANTERN_PTR lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool)(void* grad_hy, void* workspace, void* has_bias);
LANTERN_API void* (LANTERN_PTR lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor)(void* grad_hy, void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias);
LANTERN_API void* (LANTERN_PTR lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool)(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool_scalartype_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first, void* dtype, void* use_dynamic);
LANTERN_API void* (LANTERN_PTR lantern_quantized_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_quantized_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional);
LANTERN_API void* (LANTERN_PTR lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern__pack_padded_sequence_tensor_tensor_bool)(void* input, void* lengths, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool)(void* grad, void* input_size, void* batch_sizes, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt)(void* data, void* batch_sizes, void* batch_first, void* padding_value, void* total_length);
LANTERN_API void* (LANTERN_PTR lantern_set__tensor_storage)(void* self, void* source);
LANTERN_API void* (LANTERN_PTR lantern_set__tensor_storage_intt_intarrayref_intarrayref)(void* self, void* source, void* storage_offset, void* size, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_set__tensor_tensor)(void* self, void* source);
LANTERN_API void* (LANTERN_PTR lantern_set__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_set_quantizer__tensor_constquantizerptr)(void* self, void* quantizer);
LANTERN_API void* (LANTERN_PTR lantern_is_set_to_tensor_tensor)(void* self, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_masked_fill__tensor_tensor_scalar)(void* self, void* mask, void* value);
LANTERN_API void* (LANTERN_PTR lantern_masked_fill_tensor_tensor_scalar)(void* self, void* mask, void* value);
LANTERN_API void* (LANTERN_PTR lantern_masked_fill__tensor_tensor_tensor)(void* self, void* mask, void* value);
LANTERN_API void* (LANTERN_PTR lantern_masked_fill_tensor_tensor_tensor)(void* self, void* mask, void* value);
LANTERN_API void* (LANTERN_PTR lantern_masked_scatter__tensor_tensor_tensor)(void* self, void* mask, void* source);
LANTERN_API void* (LANTERN_PTR lantern_masked_scatter_tensor_tensor_tensor)(void* self, void* mask, void* source);
LANTERN_API void* (LANTERN_PTR lantern_view_tensor_intarrayref)(void* self, void* size);
LANTERN_API void* (LANTERN_PTR lantern_put__tensor_tensor_tensor_bool)(void* self, void* index, void* source, void* accumulate);
LANTERN_API void* (LANTERN_PTR lantern_index_add__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_add_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_add_tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern_index_fill__tensor_intt_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill_tensor_intt_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill__tensor_dimname_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill__tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill_tensor_dimname_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_index_fill_tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_scatter__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_scatter_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_scatter__tensor_intt_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_scatter_tensor_intt_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_scatter_tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_scatter_tensor_dimname_tensor_scalar)(void* self, void* dim, void* index, void* value);
LANTERN_API void* (LANTERN_PTR lantern_scatter_add__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_scatter_add_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_scatter_add_tensor_dimname_tensor_tensor)(void* self, void* dim, void* index, void* src);
LANTERN_API void* (LANTERN_PTR lantern_lt__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lt__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ne__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ne__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___and___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___and___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___iand___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___iand___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___or___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___or___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ior___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ior___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___xor___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___xor___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ixor___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ixor___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___lshift___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___lshift___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ilshift___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___ilshift___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___rshift___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___rshift___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___irshift___tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern___irshift___tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lgamma__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_atan2__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_tril__tensor_intt)(void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_triu__tensor_intt)(void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_digamma__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_polygamma__tensor_intt)(void* self, void* n);
LANTERN_API void* (LANTERN_PTR lantern_renorm__tensor_scalar_intt_scalar)(void* self, void* p, void* dim, void* maxnorm);
LANTERN_API void* (LANTERN_PTR lantern_pow__tensor_scalar)(void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_pow__tensor_tensor)(void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_lerp__tensor_tensor_scalar)(void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_lerp__tensor_tensor_tensor)(void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_fmod__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_fmod__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder__tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder__tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_addbmm__tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addbmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_addcdiv__tensor_tensor_tensor_scalar)(void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_random__tensor_intt_intt_generator)(void* self, void* from, void* to, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_random__tensor_intt_generator)(void* self, void* to, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_random__tensor_generator)(void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_uniform__tensor_double_double_generator)(void* self, void* from, void* to, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_normal__tensor_double_double_generator)(void* self, void* mean, void* std, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_cauchy__tensor_double_double_generator)(void* self, void* median, void* sigma, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_log_normal__tensor_double_double_generator)(void* self, void* mean, void* std, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_exponential__tensor_double_generator)(void* self, void* lambd, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_geometric__tensor_double_generator)(void* self, void* p, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_diag_out_tensor_tensor_intt)(void* out, void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_diag_tensor_intt)(void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_cross_out_tensor_tensor_tensor_intt)(void* out, void* self, void* other, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cross_tensor_tensor_intt)(void* self, void* other, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_triu_out_tensor_tensor_intt)(void* out, void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_triu_tensor_intt)(void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_tril_out_tensor_tensor_intt)(void* out, void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_tril_tensor_intt)(void* self, void* diagonal);
LANTERN_API void* (LANTERN_PTR lantern_tril_indices_intt_intt_intt_tensoroptions)(void* row, void* col, void* offset, void* options);
LANTERN_API void* (LANTERN_PTR lantern_triu_indices_intt_intt_intt_tensoroptions)(void* row, void* col, void* offset, void* options);
LANTERN_API void* (LANTERN_PTR lantern_trace_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_ne_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ne_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ne_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ne_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_eq_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_ge_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_le_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_gt_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lt_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lt_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lt_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lt_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_take_out_tensor_tensor_tensor)(void* out, void* self, void* index);
LANTERN_API void* (LANTERN_PTR lantern_take_tensor_tensor)(void* self, void* index);
LANTERN_API void* (LANTERN_PTR lantern_index_select_out_tensor_tensor_intt_tensor)(void* out, void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_index_select_tensor_intt_tensor)(void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_index_select_out_tensor_tensor_dimname_tensor)(void* out, void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_index_select_tensor_dimname_tensor)(void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_masked_select_out_tensor_tensor_tensor)(void* out, void* self, void* mask);
LANTERN_API void* (LANTERN_PTR lantern_masked_select_tensor_tensor)(void* self, void* mask);
LANTERN_API void* (LANTERN_PTR lantern_nonzero_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_nonzero_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_nonzero_numpy_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_gather_out_tensor_tensor_intt_tensor_bool)(void* out, void* self, void* dim, void* index, void* sparse_grad);
LANTERN_API void* (LANTERN_PTR lantern_gather_tensor_intt_tensor_bool)(void* self, void* dim, void* index, void* sparse_grad);
LANTERN_API void* (LANTERN_PTR lantern_gather_out_tensor_tensor_dimname_tensor_bool)(void* out, void* self, void* dim, void* index, void* sparse_grad);
LANTERN_API void* (LANTERN_PTR lantern_gather_tensor_dimname_tensor_bool)(void* self, void* dim, void* index, void* sparse_grad);
LANTERN_API void* (LANTERN_PTR lantern__gather_sparse_backward_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* grad);
LANTERN_API void* (LANTERN_PTR lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar)(void* out, void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_addcmul_tensor_tensor_tensor_scalar)(void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_addcmul__tensor_tensor_tensor_scalar)(void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar)(void* out, void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_addcdiv_tensor_tensor_tensor_scalar)(void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_lstsq_out_tensor_tensor_tensor_tensor)(void* X, void* qr, void* self, void* A);
LANTERN_API void* (LANTERN_PTR lantern_lstsq_tensor_tensor)(void* self, void* A);
LANTERN_API void* (LANTERN_PTR lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool)(void* X, void* M, void* self, void* A, void* upper, void* transpose, void* unitriangular);
LANTERN_API void* (LANTERN_PTR lantern_triangular_solve_tensor_tensor_bool_bool_bool)(void* self, void* A, void* upper, void* transpose, void* unitriangular);
LANTERN_API void* (LANTERN_PTR lantern__triangular_solve_helper_tensor_tensor_bool_bool_bool)(void* self, void* A, void* upper, void* transpose, void* unitriangular);
LANTERN_API void* (LANTERN_PTR lantern_symeig_out_tensor_tensor_tensor_bool_bool)(void* e, void* V, void* self, void* eigenvectors, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_symeig_tensor_bool_bool)(void* self, void* eigenvectors, void* upper);
LANTERN_API void* (LANTERN_PTR lantern__symeig_helper_tensor_bool_bool)(void* self, void* eigenvectors, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_eig_out_tensor_tensor_tensor_bool)(void* e, void* v, void* self, void* eigenvectors);
LANTERN_API void* (LANTERN_PTR lantern_eig_tensor_bool)(void* self, void* eigenvectors);
LANTERN_API void* (LANTERN_PTR lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool)(void* U, void* S, void* V, void* self, void* some, void* compute_uv);
LANTERN_API void* (LANTERN_PTR lantern_svd_tensor_bool_bool)(void* self, void* some, void* compute_uv);
LANTERN_API void* (LANTERN_PTR lantern__svd_helper_tensor_bool_bool)(void* self, void* some, void* compute_uv);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_out_tensor_tensor_bool)(void* out, void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_tensor_bool)(void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern__cholesky_helper_tensor_bool)(void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_solve_out_tensor_tensor_tensor_bool)(void* out, void* self, void* input2, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_solve_tensor_tensor_bool)(void* self, void* input2, void* upper);
LANTERN_API void* (LANTERN_PTR lantern__cholesky_solve_helper_tensor_tensor_bool)(void* self, void* A, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_solve_tensor_tensor)(void* self, void* A);
LANTERN_API void* (LANTERN_PTR lantern_solve_out_tensor_tensor_tensor_tensor)(void* solution, void* lu, void* self, void* A);
LANTERN_API void* (LANTERN_PTR lantern__solve_helper_tensor_tensor)(void* self, void* A);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_inverse_out_tensor_tensor_bool)(void* out, void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_inverse_tensor_bool)(void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_qr_out_tensor_tensor_tensor_bool)(void* Q, void* R, void* self, void* some);
LANTERN_API void* (LANTERN_PTR lantern_qr_tensor_bool)(void* self, void* some);
LANTERN_API void* (LANTERN_PTR lantern__qr_helper_tensor_bool)(void* self, void* some);
LANTERN_API void* (LANTERN_PTR lantern_geqrf_out_tensor_tensor_tensor)(void* a, void* tau, void* self);
LANTERN_API void* (LANTERN_PTR lantern_geqrf_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_orgqr_out_tensor_tensor_tensor)(void* out, void* self, void* input2);
LANTERN_API void* (LANTERN_PTR lantern_orgqr_tensor_tensor)(void* self, void* input2);
LANTERN_API void* (LANTERN_PTR lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool)(void* out, void* self, void* input2, void* input3, void* left, void* transpose);
LANTERN_API void* (LANTERN_PTR lantern_ormqr_tensor_tensor_tensor_bool_bool)(void* self, void* input2, void* input3, void* left, void* transpose);
LANTERN_API void* (LANTERN_PTR lantern__lu_with_info_tensor_bool_bool)(void* self, void* pivot, void* check_errors);
LANTERN_API void* (LANTERN_PTR lantern_lu_solve_out_tensor_tensor_tensor_tensor)(void* out, void* self, void* LU_data, void* LU_pivots);
LANTERN_API void* (LANTERN_PTR lantern_lu_solve_tensor_tensor_tensor)(void* self, void* LU_data, void* LU_pivots);
LANTERN_API void* (LANTERN_PTR lantern__lu_solve_helper_tensor_tensor_tensor)(void* self, void* LU_data, void* LU_pivots);
LANTERN_API void* (LANTERN_PTR lantern_multinomial_out_tensor_tensor_intt_bool_generator)(void* out, void* self, void* num_samples, void* replacement, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_multinomial_tensor_intt_bool_generator)(void* self, void* num_samples, void* replacement, void* generator);
LANTERN_API void* (LANTERN_PTR lantern__multinomial_alias_setup_tensor)(void* probs);
LANTERN_API void* (LANTERN_PTR lantern__multinomial_alias_draw_tensor_tensor_intt_generator)(void* J, void* q, void* num_samples, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_lgamma_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_lgamma_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_digamma_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_digamma_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_polygamma_out_tensor_intt_tensor)(void* out, void* n, void* self);
LANTERN_API void* (LANTERN_PTR lantern_polygamma_intt_tensor)(void* n, void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfinv_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfinv__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_erfinv_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_sign_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sign__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sign_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_dist_tensor_tensor_scalar)(void* self, void* other, void* p);
LANTERN_API void* (LANTERN_PTR lantern_atan2_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_atan2_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_lerp_out_tensor_tensor_tensor_scalar)(void* out, void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_lerp_out_tensor_tensor_tensor_tensor)(void* out, void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_lerp_tensor_tensor_scalar)(void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_lerp_tensor_tensor_tensor)(void* self, void* end, void* weight);
LANTERN_API void* (LANTERN_PTR lantern_histc_out_tensor_tensor_intt_scalar_scalar)(void* out, void* self, void* bins, void* min, void* max);
LANTERN_API void* (LANTERN_PTR lantern_histc_tensor_intt_scalar_scalar)(void* self, void* bins, void* min, void* max);
LANTERN_API void* (LANTERN_PTR lantern_fmod_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_fmod_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_fmod_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_fmod_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder_out_tensor_tensor_scalar)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder_tensor_scalar)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_remainder_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_min_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_min_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_min_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_max_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_max_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_max_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_median_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_sort_out_tensor_tensor_tensor_intt_bool)(void* values, void* indices, void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_sort_tensor_intt_bool)(void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_sort_out_tensor_tensor_tensor_dimname_bool)(void* values, void* indices, void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_sort_tensor_dimname_bool)(void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_argsort_tensor_intt_bool)(void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_argsort_tensor_dimname_bool)(void* self, void* dim, void* descending);
LANTERN_API void* (LANTERN_PTR lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool)(void* values, void* indices, void* self, void* k, void* dim, void* largest, void* sorted);
LANTERN_API void* (LANTERN_PTR lantern_topk_tensor_intt_intt_bool_bool)(void* self, void* k, void* dim, void* largest, void* sorted);
LANTERN_API void* (LANTERN_PTR lantern_all_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_any_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_renorm_out_tensor_tensor_scalar_intt_scalar)(void* out, void* self, void* p, void* dim, void* maxnorm);
LANTERN_API void* (LANTERN_PTR lantern_renorm_tensor_scalar_intt_scalar)(void* self, void* p, void* dim, void* maxnorm);
LANTERN_API void* (LANTERN_PTR lantern_unfold_tensor_intt_intt_intt)(void* self, void* dimension, void* size, void* step);
LANTERN_API void* (LANTERN_PTR lantern_equal_tensor_tensor)(void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_pow_out_tensor_tensor_tensor)(void* out, void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_pow_tensor_tensor)(void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_pow_out_tensor_scalar_tensor)(void* out, void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_pow_scalar_tensor)(void* self, void* exponent);
LANTERN_API void* (LANTERN_PTR lantern_normal_out_tensor_tensor_double_generator)(void* out, void* mean, void* std, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_normal_out_tensor_double_tensor_generator)(void* out, void* mean, void* std, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_normal_out_tensor_tensor_tensor_generator)(void* out, void* mean, void* std, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_normal_out_tensor_double_double_intarrayref_generator)(void* out, void* mean, void* std, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_alias_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__addr_tensor_tensor_tensor_scalar_scalar)(void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__addr__tensor_tensor_tensor_scalar_scalar)(void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__addr_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__index_copy__tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* source);
LANTERN_API void* (LANTERN_PTR lantern__cumsum_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__cumsum_out_tensor_tensor_intt)(void* out, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__cumprod_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__cumprod_out_tensor_tensor_intt)(void* out, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__var_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern__std_tensor_bool)(void* self, void* unbiased);
LANTERN_API void* (LANTERN_PTR lantern__cat_tensorlist_intt)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__cat_out_tensor_tensorlist_intt)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__mode_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern__mode_out_tensor_tensor_tensor_intt_bool)(void* values, void* indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern__max_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern__max_out_tensor_tensor_tensor_intt_bool)(void* max, void* max_indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern__min_tensor_intt_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern__min_out_tensor_tensor_tensor_intt_bool)(void* min, void* min_indices, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_tensor_tensor_tensor_intt)(void* self, void* target, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_mse_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_mse_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_mse_loss_backward_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_l1_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_l1_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_l1_loss_backward_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt)(void* out, void* self, void* target, void* p, void* margin, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt)(void* self, void* target, void* p, void* margin, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt)(void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt)(void* output, void* is_target, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_forward_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor)(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* is_target);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor)(void* grad_output, void* self, void* target, void* reduction, void* is_target);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt)(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt)(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor)(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor)(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt)(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt)(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor)(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor)(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight);
LANTERN_API void* (LANTERN_PTR lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_smooth_l1_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_soft_margin_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_soft_margin_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_elu_out_tensor_tensor_scalar_scalar_scalar)(void* out, void* self, void* alpha, void* scale, void* input_scale);
LANTERN_API void* (LANTERN_PTR lantern_elu_tensor_scalar_scalar_scalar)(void* self, void* alpha, void* scale, void* input_scale);
LANTERN_API void* (LANTERN_PTR lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_tensor)(void* grad_input, void* grad_output, void* alpha, void* scale, void* input_scale, void* output);
LANTERN_API void* (LANTERN_PTR lantern_elu_backward_tensor_scalar_scalar_scalar_tensor)(void* grad_output, void* alpha, void* scale, void* input_scale, void* output);
LANTERN_API void* (LANTERN_PTR lantern_elu__tensor_scalar_scalar_scalar)(void* self, void* alpha, void* scale, void* input_scale);
LANTERN_API void* (LANTERN_PTR lantern_glu_out_tensor_tensor_intt)(void* out, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_glu_tensor_intt)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_glu_backward_out_tensor_tensor_tensor_intt)(void* grad_input, void* grad_output, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_glu_backward_tensor_tensor_intt)(void* grad_output, void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_hardtanh_out_tensor_tensor_scalar_scalar)(void* out, void* self, void* min_val, void* max_val);
LANTERN_API void* (LANTERN_PTR lantern_hardtanh_tensor_scalar_scalar)(void* self, void* min_val, void* max_val);
LANTERN_API void* (LANTERN_PTR lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar)(void* grad_input, void* grad_output, void* self, void* min_val, void* max_val);
LANTERN_API void* (LANTERN_PTR lantern_hardtanh_backward_tensor_tensor_scalar_scalar)(void* grad_output, void* self, void* min_val, void* max_val);
LANTERN_API void* (LANTERN_PTR lantern_hardtanh__tensor_scalar_scalar)(void* self, void* min_val, void* max_val);
LANTERN_API void* (LANTERN_PTR lantern_leaky_relu_out_tensor_tensor_scalar)(void* out, void* self, void* negative_slope);
LANTERN_API void* (LANTERN_PTR lantern_leaky_relu_tensor_scalar)(void* self, void* negative_slope);
LANTERN_API void* (LANTERN_PTR lantern_leaky_relu_backward_out_tensor_tensor_tensor_scalar)(void* grad_input, void* grad_output, void* self, void* negative_slope);
LANTERN_API void* (LANTERN_PTR lantern_leaky_relu_backward_tensor_tensor_scalar)(void* grad_output, void* self, void* negative_slope);
LANTERN_API void* (LANTERN_PTR lantern_leaky_relu__tensor_scalar)(void* self, void* negative_slope);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_forward_out_tensor_tensor_tensor)(void* output, void* buffer, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_forward_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* self, void* buffer);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_backward_tensor_tensor_tensor)(void* grad_output, void* self, void* buffer);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator)(void* out, void* self, void* noise, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator)(void* self, void* noise, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_with_noise_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_bool)(void* grad_input, void* grad_output, void* self, void* noise, void* lower, void* upper, void* training);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool)(void* grad_output, void* self, void* noise, void* lower, void* upper, void* training);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator)(void* self, void* noise, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_softplus_out_tensor_tensor_scalar_scalar)(void* out, void* self, void* beta, void* threshold);
LANTERN_API void* (LANTERN_PTR lantern_softplus_tensor_scalar_scalar)(void* self, void* beta, void* threshold);
LANTERN_API void* (LANTERN_PTR lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor)(void* grad_input, void* grad_output, void* self, void* beta, void* threshold, void* output);
LANTERN_API void* (LANTERN_PTR lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor)(void* grad_output, void* self, void* beta, void* threshold, void* output);
LANTERN_API void* (LANTERN_PTR lantern_softshrink_out_tensor_tensor_scalar)(void* out, void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_softshrink_tensor_scalar)(void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_softshrink_backward_out_tensor_tensor_tensor_scalar)(void* grad_input, void* grad_output, void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_softshrink_backward_tensor_tensor_scalar)(void* grad_output, void* self, void* lambd);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref)(void* out, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern__adaptive_avg_pool2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern__adaptive_avg_pool2d_backward_tensor_tensor)(void* grad_output, void* self);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref)(void* out, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool3d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* self);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool3d_backward_tensor_tensor)(void* grad_output, void* self);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref)(void* out, void* indices, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* self, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor)(void* grad_output, void* self, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref)(void* out, void* indices, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool3d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* self, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor)(void* grad_output, void* self, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor)(void* self, void* kernel_size, void* output_size, void* random_samples);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor)(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor)(void* self, void* kernel_size, void* output_size, void* random_samples);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor)(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref)(void* out, void* self, void* indices, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool2d_tensor_tensor_intarrayref)(void* self, void* indices, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* indices, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref)(void* grad_output, void* self, void* indices, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* indices, void* output_size, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* indices, void* output_size, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)(void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad1d_out_tensor_tensor_intarrayref)(void* out, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad1d_tensor_intarrayref)(void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad1d_backward_tensor_tensor_intarrayref)(void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad2d_out_tensor_tensor_intarrayref)(void* out, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad2d_tensor_intarrayref)(void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_reflection_pad2d_backward_tensor_tensor_intarrayref)(void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad1d_out_tensor_tensor_intarrayref)(void* out, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad1d_tensor_intarrayref)(void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad1d_backward_tensor_tensor_intarrayref)(void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad2d_out_tensor_tensor_intarrayref)(void* out, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad2d_tensor_intarrayref)(void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad2d_backward_tensor_tensor_intarrayref)(void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad3d_out_tensor_tensor_intarrayref)(void* out, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad3d_tensor_intarrayref)(void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref)(void* grad_input, void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_replication_pad3d_backward_tensor_tensor_intarrayref)(void* grad_output, void* self, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_linear1d_tensor_intarrayref_bool)(void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool)(void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bilinear2d_tensor_intarrayref_bool)(void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool)(void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bicubic2d_tensor_intarrayref_bool)(void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool)(void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_trilinear3d_tensor_intarrayref_bool)(void* self, void* output_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool)(void* grad_output, void* output_size, void* input_size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest1d_out_tensor_tensor_intarrayref)(void* out, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest1d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref)(void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest2d_out_tensor_tensor_intarrayref)(void* out, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref)(void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest3d_out_tensor_tensor_intarrayref)(void* out, void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest3d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref)(void* grad_output, void* output_size, void* input_size);
LANTERN_API void* (LANTERN_PTR lantern_sigmoid_backward_out_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* output);
LANTERN_API void* (LANTERN_PTR lantern_sigmoid_backward_tensor_tensor)(void* grad_output, void* output);
LANTERN_API void* (LANTERN_PTR lantern_tanh_backward_out_tensor_tensor_tensor)(void* grad_input, void* grad_output, void* output);
LANTERN_API void* (LANTERN_PTR lantern_tanh_backward_tensor_tensor)(void* grad_output, void* output);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor)(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor)(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor)(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* grad_input, void* grad_weight, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor)(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask);
LANTERN_API void* (LANTERN_PTR lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)(void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref)(void* self, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)(void* grad_input, void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride);
LANTERN_API void* (LANTERN_PTR lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)(void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride);
/* Autogen Headers -- End */
  
#ifdef __cplusplus
}
#endif

#ifndef LANTERN_HEADERS_ONLY

#include <string>

inline const char* pathSeparator()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
  
inline const char* libraryName()
{
#ifdef __APPLE__
  return "liblantern.dylib";
#else
#ifdef _WIN32
  return "lantern.dll";
#else
  return "liblantern.so";
#endif
#endif
}

void* pLibrary = NULL;

#define LOAD_SYMBOL(name)                                     \
if (!laternLoadSymbol(pLibrary, #name, (void**) &name, pError))     \
  return false;

void lanternLoadError(std::string* pError)
{
#ifdef _WIN32
  LPVOID lpMsgBuf;
  DWORD dw = ::GetLastError();
  
  DWORD length = ::FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      dw,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &lpMsgBuf,
      0, NULL );
  
  if (length != 0)
  {
    std::string msg((LPTSTR)lpMsgBuf);
    LocalFree(lpMsgBuf);
    pError->assign(msg);
  }
  else
  {
    pError->assign("Unknown error");
  }
#else
  const char* msg = ::dlerror();
  if (msg != NULL)
    pError->assign(msg);
  else
    pError->assign("Unknown error");
#endif
}

bool lanternLoadLibrary(const std::string& libPath, std::string* pError)
{
  pLibrary = NULL;

  char lastLibChar = libPath.at(libPath.size() - 1);
  std::string separator = (lastLibChar == '/' || lastLibChar == '\\') ? "" : pathSeparator();
  std::string libFile = libPath + separator + libraryName();

#ifdef _WIN32
  
  typedef DLL_DIRECTORY_COOKIE(WINAPI * PAddDllDirectory)(PCWSTR);
  HMODULE hKernel = ::GetModuleHandle("kernel32.dll");

  if (hKernel == NULL) {
    lanternLoadError(pError);
    *pError = "Get Kernel - " + *pError;
    return false;
  }
  
  PAddDllDirectory add_dll_directory = (PAddDllDirectory)::GetProcAddress(hKernel, "AddDllDirectory");
  
  if (add_dll_directory != NULL) {
    std::wstring libPathWStr = std::wstring(libPath.begin(), libPath.end());
    DLL_DIRECTORY_COOKIE cookie = add_dll_directory(libPathWStr.c_str());

    if (cookie == NULL) {
      lanternLoadError(pError);
      *pError = "Add Dll Directory - " + *pError;
      return false;
    }
  }

  pLibrary = (void*)::LoadLibraryEx(libFile.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
  pLibrary = ::dlopen(libFile.c_str(), RTLD_NOW|RTLD_GLOBAL);
#endif
  if (pLibrary == NULL)
  {
    lanternLoadError(pError);
    *pError = libFile + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool laternLoadSymbol(void* pLib, const std::string& name, void** ppSymbol, std::string* pError)
{
  *ppSymbol = NULL;
#ifdef _WIN32
  *ppSymbol = (void*)::GetProcAddress((HINSTANCE)pLib, name.c_str());
#else
  *ppSymbol = ::dlsym(pLib, name.c_str());
#endif
  if (*ppSymbol == NULL)
  {
    lanternLoadError(pError);
    *pError = name + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool laternCloseLibrary(void* pLib, std::string* pError)
{
#ifdef _WIN32
  if (!::FreeLibrary((HMODULE)pLib))
#else
  if (::dlclose(pLib) != 0)
#endif
  {
    lanternLoadError(pError);
    return false;
  }
  else
  {
    return true;
  }
}

bool lanternInit(const std::string& libPath, std::string* pError)
{
  if (!lanternLoadLibrary(libPath, pError))
    return false;
  
  LOAD_SYMBOL(lanternTest);

  /* Autogen Symbols -- Start */
  LOAD_SYMBOL(lantern__cast_byte_tensor_bool)
  LOAD_SYMBOL(lantern__cast_char_tensor_bool)
  LOAD_SYMBOL(lantern__cast_double_tensor_bool)
  LOAD_SYMBOL(lantern__cast_float_tensor_bool)
  LOAD_SYMBOL(lantern__cast_int_tensor_bool)
  LOAD_SYMBOL(lantern__cast_long_tensor_bool)
  LOAD_SYMBOL(lantern__cast_short_tensor_bool)
  LOAD_SYMBOL(lantern__cast_half_tensor_bool)
  LOAD_SYMBOL(lantern_backward_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern_set_data_tensor_tensor)
  LOAD_SYMBOL(lantern_data_tensor)
  LOAD_SYMBOL(lantern_is_leaf_tensor)
  LOAD_SYMBOL(lantern_output_nr_tensor)
  LOAD_SYMBOL(lantern__version_tensor)
  LOAD_SYMBOL(lantern_rename__tensor_dimnamelist)
  LOAD_SYMBOL(lantern_rename_tensor_dimnamelist)
  LOAD_SYMBOL(lantern_align_to_tensor_dimnamelist)
  LOAD_SYMBOL(lantern_align_as_tensor_tensor)
  LOAD_SYMBOL(lantern_align_tensors_tensorlist)
  LOAD_SYMBOL(lantern_refine_names_tensor_dimnamelist)
  LOAD_SYMBOL(lantern_unflatten_tensor_dimname_intarrayref_dimnamelist)
  LOAD_SYMBOL(lantern_unflatten_tensor_intt_intarrayref_dimnamelist)
  LOAD_SYMBOL(lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_bool_bool)
  LOAD_SYMBOL(lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor)
  LOAD_SYMBOL(lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions)
  LOAD_SYMBOL(lantern__debug_has_internal_overlap_tensor)
  LOAD_SYMBOL(lantern__fused_dropout_tensor_double_generator)
  LOAD_SYMBOL(lantern__masked_scale_tensor_tensor_double)
  LOAD_SYMBOL(lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype)
  LOAD_SYMBOL(lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt)
  LOAD_SYMBOL(lantern__sobol_engine_scramble__tensor_tensor_intt)
  LOAD_SYMBOL(lantern__sobol_engine_initialize_state__tensor_intt)
  LOAD_SYMBOL(lantern__reshape_from_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__shape_as_tensor_tensor)
  LOAD_SYMBOL(lantern_dropout_tensor_double_bool)
  LOAD_SYMBOL(lantern_dropout__tensor_double_bool)
  LOAD_SYMBOL(lantern_feature_dropout_tensor_double_bool)
  LOAD_SYMBOL(lantern_feature_dropout__tensor_double_bool)
  LOAD_SYMBOL(lantern_alpha_dropout_tensor_double_bool)
  LOAD_SYMBOL(lantern_alpha_dropout__tensor_double_bool)
  LOAD_SYMBOL(lantern_feature_alpha_dropout_tensor_double_bool)
  LOAD_SYMBOL(lantern_feature_alpha_dropout__tensor_double_bool)
  LOAD_SYMBOL(lantern_abs_tensor)
  LOAD_SYMBOL(lantern_abs__tensor)
  LOAD_SYMBOL(lantern_abs_out_tensor_tensor)
  LOAD_SYMBOL(lantern_acos_tensor)
  LOAD_SYMBOL(lantern_acos__tensor)
  LOAD_SYMBOL(lantern_acos_out_tensor_tensor)
  LOAD_SYMBOL(lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_adaptive_avg_pool1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_max_pool1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_add_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_add__tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_add_out_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_add_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_add__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmv_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmv__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addr_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addr__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_affine_grid_generator_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_affine_grid_generator_backward_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_all_tensor_intt_bool)
  LOAD_SYMBOL(lantern_all_out_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_all_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_all_out_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_allclose_tensor_tensor_double_double_bool)
  LOAD_SYMBOL(lantern_any_tensor_intt_bool)
  LOAD_SYMBOL(lantern_any_out_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_any_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_any_out_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_arange_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_scalar_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_out_tensor_scalar)
  LOAD_SYMBOL(lantern_arange_out_tensor_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern__dim_arange_tensor_intt)
  LOAD_SYMBOL(lantern_argmax_tensor_intt_bool)
  LOAD_SYMBOL(lantern_argmin_tensor_intt_bool)
  LOAD_SYMBOL(lantern_as_strided_tensor_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_as_strided__tensor_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_asin_tensor)
  LOAD_SYMBOL(lantern_asin__tensor)
  LOAD_SYMBOL(lantern_asin_out_tensor_tensor)
  LOAD_SYMBOL(lantern_atan_tensor)
  LOAD_SYMBOL(lantern_atan__tensor)
  LOAD_SYMBOL(lantern_atan_out_tensor_tensor)
  LOAD_SYMBOL(lantern_baddbmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_baddbmm__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_bartlett_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_bartlett_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)
  LOAD_SYMBOL(lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)
  LOAD_SYMBOL(lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool)
  LOAD_SYMBOL(lantern_bernoulli_tensor_generator)
  LOAD_SYMBOL(lantern_bernoulli_out_tensor_tensor_generator)
  LOAD_SYMBOL(lantern_bernoulli__tensor_tensor_generator)
  LOAD_SYMBOL(lantern_bernoulli__tensor_double_generator)
  LOAD_SYMBOL(lantern_bernoulli_tensor_double_generator)
  LOAD_SYMBOL(lantern_bilinear_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_bincount_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_bitwise_not_tensor)
  LOAD_SYMBOL(lantern_bitwise_not__tensor)
  LOAD_SYMBOL(lantern_bitwise_not_out_tensor_tensor)
  LOAD_SYMBOL(lantern_logical_not_tensor)
  LOAD_SYMBOL(lantern_logical_not__tensor)
  LOAD_SYMBOL(lantern_logical_not_out_tensor_tensor)
  LOAD_SYMBOL(lantern_logical_xor_tensor_tensor)
  LOAD_SYMBOL(lantern_logical_xor__tensor_tensor)
  LOAD_SYMBOL(lantern_logical_xor_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_blackman_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_blackman_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_bmm_tensor_tensor)
  LOAD_SYMBOL(lantern_bmm_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_broadcast_tensors_tensorlist)
  LOAD_SYMBOL(lantern_cat_tensorlist_intt)
  LOAD_SYMBOL(lantern_cat_out_tensor_tensorlist_intt)
  LOAD_SYMBOL(lantern_cat_tensorlist_dimname)
  LOAD_SYMBOL(lantern_cat_out_tensor_tensorlist_dimname)
  LOAD_SYMBOL(lantern_ceil_tensor)
  LOAD_SYMBOL(lantern_ceil__tensor)
  LOAD_SYMBOL(lantern_ceil_out_tensor_tensor)
  LOAD_SYMBOL(lantern_chain_matmul_tensorlist)
  LOAD_SYMBOL(lantern_chunk_tensor_intt_intt)
  LOAD_SYMBOL(lantern_clamp_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_clamp__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_clamp_out_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_clamp_max_tensor_scalar)
  LOAD_SYMBOL(lantern_clamp_max__tensor_scalar)
  LOAD_SYMBOL(lantern_clamp_max_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_clamp_min_tensor_scalar)
  LOAD_SYMBOL(lantern_clamp_min__tensor_scalar)
  LOAD_SYMBOL(lantern_clamp_min_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_cudnn_is_acceptable_tensor)
  LOAD_SYMBOL(lantern_constant_pad_nd_tensor_intarrayref_scalar)
  LOAD_SYMBOL(lantern_contiguous_tensor_memoryformat)
  LOAD_SYMBOL(lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)
  LOAD_SYMBOL(lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)
  LOAD_SYMBOL(lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool)
  LOAD_SYMBOL(lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool)
  LOAD_SYMBOL(lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref)
  LOAD_SYMBOL(lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv_tbc_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_copy__tensor_tensor_bool)
  LOAD_SYMBOL(lantern__copy_from_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_cos_tensor)
  LOAD_SYMBOL(lantern_cos__tensor)
  LOAD_SYMBOL(lantern_cos_out_tensor_tensor)
  LOAD_SYMBOL(lantern_cosh_tensor)
  LOAD_SYMBOL(lantern_cosh__tensor)
  LOAD_SYMBOL(lantern_cosh_out_tensor_tensor)
  LOAD_SYMBOL(lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt)
  LOAD_SYMBOL(lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt)
  LOAD_SYMBOL(lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)
  LOAD_SYMBOL(lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double)
  LOAD_SYMBOL(lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_bias_tensor)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_bias_tensor)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_cudnn_grid_sampler_tensor_tensor)
  LOAD_SYMBOL(lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_cumsum_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumsum_out_tensor_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumsum_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_cumsum_out_tensor_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_cumprod_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumprod_out_tensor_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumprod_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_cumprod_out_tensor_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool)
  LOAD_SYMBOL(lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool)
  LOAD_SYMBOL(lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_det_tensor)
  LOAD_SYMBOL(lantern_diag_embed_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_diagflat_tensor_intt)
  LOAD_SYMBOL(lantern_diagonal_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_fill_diagonal__tensor_scalar_bool)
  LOAD_SYMBOL(lantern_div_tensor_tensor)
  LOAD_SYMBOL(lantern_div__tensor_tensor)
  LOAD_SYMBOL(lantern_div_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_div_tensor_scalar)
  LOAD_SYMBOL(lantern_div__tensor_scalar)
  LOAD_SYMBOL(lantern_dot_tensor_tensor)
  LOAD_SYMBOL(lantern_dot_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_einsum_stdstring_tensorlist)
  LOAD_SYMBOL(lantern_embedding_tensor_tensor_intt_bool_bool)
  LOAD_SYMBOL(lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool)
  LOAD_SYMBOL(lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_embedding_renorm__tensor_tensor_double_double)
  LOAD_SYMBOL(lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_empty_intarrayref_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_new_empty_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_new_full_tensor_intarrayref_scalar_tensoroptions)
  LOAD_SYMBOL(lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat)
  LOAD_SYMBOL(lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_resize__tensor_intarrayref)
  LOAD_SYMBOL(lantern_empty_out_tensor_intarrayref_memoryformat)
  LOAD_SYMBOL(lantern_empty_like_tensor)
  LOAD_SYMBOL(lantern_empty_like_tensor_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_empty_strided_intarrayref_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_erf_tensor)
  LOAD_SYMBOL(lantern_erf__tensor)
  LOAD_SYMBOL(lantern_erf_out_tensor_tensor)
  LOAD_SYMBOL(lantern_erfc_tensor)
  LOAD_SYMBOL(lantern_erfc__tensor)
  LOAD_SYMBOL(lantern_erfc_out_tensor_tensor)
  LOAD_SYMBOL(lantern_exp_tensor)
  LOAD_SYMBOL(lantern_exp__tensor)
  LOAD_SYMBOL(lantern_exp_out_tensor_tensor)
  LOAD_SYMBOL(lantern_expm1_tensor)
  LOAD_SYMBOL(lantern_expm1__tensor)
  LOAD_SYMBOL(lantern_expm1_out_tensor_tensor)
  LOAD_SYMBOL(lantern_expand_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_expand_as_tensor_tensor)
  LOAD_SYMBOL(lantern_eye_intt_tensoroptions)
  LOAD_SYMBOL(lantern_eye_intt_intt_tensoroptions)
  LOAD_SYMBOL(lantern_eye_out_tensor_intt)
  LOAD_SYMBOL(lantern_eye_out_tensor_intt_intt)
  LOAD_SYMBOL(lantern_flatten_tensor_intt_intt)
  LOAD_SYMBOL(lantern_flatten_tensor_intt_intt_dimname)
  LOAD_SYMBOL(lantern_flatten_tensor_dimname_dimname_dimname)
  LOAD_SYMBOL(lantern_flatten_tensor_dimnamelist_dimname)
  LOAD_SYMBOL(lantern_fill__tensor_scalar)
  LOAD_SYMBOL(lantern_fill__tensor_tensor)
  LOAD_SYMBOL(lantern_floor_tensor)
  LOAD_SYMBOL(lantern_floor__tensor)
  LOAD_SYMBOL(lantern_floor_out_tensor_tensor)
  LOAD_SYMBOL(lantern_frac_tensor)
  LOAD_SYMBOL(lantern_frac__tensor)
  LOAD_SYMBOL(lantern_frac_out_tensor_tensor)
  LOAD_SYMBOL(lantern_full_intarrayref_scalar_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_full_intarrayref_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_full_out_tensor_intarrayref_scalar)
  LOAD_SYMBOL(lantern_full_like_tensor_scalar)
  LOAD_SYMBOL(lantern_full_like_tensor_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_from_file_stdstring_bool_intt_tensoroptions)
  LOAD_SYMBOL(lantern_grid_sampler_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_hann_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_hann_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_double_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_double_double_tensoroptions)
  LOAD_SYMBOL(lantern_hinge_embedding_loss_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_ger_tensor_tensor)
  LOAD_SYMBOL(lantern_ger_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_group_norm_tensor_intt_tensor_tensor_double_bool)
  LOAD_SYMBOL(lantern_fft_tensor_intt_bool)
  LOAD_SYMBOL(lantern_ifft_tensor_intt_bool)
  LOAD_SYMBOL(lantern_rfft_tensor_intt_bool_bool)
  LOAD_SYMBOL(lantern_irfft_tensor_intt_bool_bool_intarrayref)
  LOAD_SYMBOL(lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_bool_bool_intarrayref)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_size_intt)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_max_size_intt)
  LOAD_SYMBOL(lantern__cufft_set_plan_cache_max_size_intt_intt)
  LOAD_SYMBOL(lantern__cufft_clear_plan_cache_intt)
  LOAD_SYMBOL(lantern_index_tensor_tensorlist)
  LOAD_SYMBOL(lantern_index_copy__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_copy_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_copy__tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_index_copy_tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_index_put__tensor_tensorlist_tensor_bool)
  LOAD_SYMBOL(lantern_index_put_tensor_tensorlist_tensor_bool)
  LOAD_SYMBOL(lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool)
  LOAD_SYMBOL(lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)
  LOAD_SYMBOL(lantern_inverse_tensor)
  LOAD_SYMBOL(lantern_inverse_out_tensor_tensor)
  LOAD_SYMBOL(lantern__inverse_helper_tensor)
  LOAD_SYMBOL(lantern_isclose_tensor_tensor_double_double_bool)
  LOAD_SYMBOL(lantern_isnan_tensor)
  LOAD_SYMBOL(lantern_is_distributed_tensor)
  LOAD_SYMBOL(lantern_is_floating_point_tensor)
  LOAD_SYMBOL(lantern_is_complex_tensor)
  LOAD_SYMBOL(lantern_is_nonzero_tensor)
  LOAD_SYMBOL(lantern_is_same_size_tensor_tensor)
  LOAD_SYMBOL(lantern_is_signed_tensor)
  LOAD_SYMBOL(lantern_kl_div_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_kl_div_backward_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_kthvalue_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_kthvalue_tensor_intt_dimname_bool)
  LOAD_SYMBOL(lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool)
  LOAD_SYMBOL(lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool)
  LOAD_SYMBOL(lantern_native_layer_norm_tensor_tensor_tensor_intt_intt_double)
  LOAD_SYMBOL(lantern_native_layer_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool)
  LOAD_SYMBOL(lantern_native_layer_norm_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool)
  LOAD_SYMBOL(lantern_linear_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_mkldnn_linear_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_quantize_weight_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_gemm_matrix_fp16_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt)
  LOAD_SYMBOL(lantern_linspace_scalar_scalar_intt_tensoroptions)
  LOAD_SYMBOL(lantern_linspace_out_tensor_scalar_scalar_intt)
  LOAD_SYMBOL(lantern_log_tensor)
  LOAD_SYMBOL(lantern_log__tensor)
  LOAD_SYMBOL(lantern_log_out_tensor_tensor)
  LOAD_SYMBOL(lantern_log10_tensor)
  LOAD_SYMBOL(lantern_log10__tensor)
  LOAD_SYMBOL(lantern_log10_out_tensor_tensor)
  LOAD_SYMBOL(lantern_log1p_tensor)
  LOAD_SYMBOL(lantern_log1p__tensor)
  LOAD_SYMBOL(lantern_log1p_out_tensor_tensor)
  LOAD_SYMBOL(lantern_log2_tensor)
  LOAD_SYMBOL(lantern_log2__tensor)
  LOAD_SYMBOL(lantern_log2_out_tensor_tensor)
  LOAD_SYMBOL(lantern_logdet_tensor)
  LOAD_SYMBOL(lantern_logspace_scalar_scalar_intt_double_tensoroptions)
  LOAD_SYMBOL(lantern_logspace_out_tensor_scalar_scalar_intt_double)
  LOAD_SYMBOL(lantern_log_softmax_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_log_softmax_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern__log_softmax_tensor_intt_bool)
  LOAD_SYMBOL(lantern__log_softmax_backward_data_tensor_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_logsumexp_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_logsumexp_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_logsumexp_tensor_dimnamelist_bool)
  LOAD_SYMBOL(lantern_logsumexp_out_tensor_tensor_dimnamelist_bool)
  LOAD_SYMBOL(lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_matmul_tensor_tensor)
  LOAD_SYMBOL(lantern_matmul_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_matrix_rank_tensor_double_bool)
  LOAD_SYMBOL(lantern_matrix_rank_tensor_bool)
  LOAD_SYMBOL(lantern_matrix_power_tensor_intt)
  LOAD_SYMBOL(lantern_max_tensor_intt_bool)
  LOAD_SYMBOL(lantern_max_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_max_values_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_max_out_tensor_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_max_values_tensor_dimnamelist_bool)
  LOAD_SYMBOL(lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_mean_tensor_scalartype)
  LOAD_SYMBOL(lantern_mean_tensor_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_mean_tensor_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_median_tensor_intt_bool)
  LOAD_SYMBOL(lantern_median_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_median_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_median_out_tensor_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_min_tensor_intt_bool)
  LOAD_SYMBOL(lantern_min_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_min_values_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_min_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_min_out_tensor_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_min_values_tensor_dimnamelist_bool)
  LOAD_SYMBOL(lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool)
  LOAD_SYMBOL(lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)
  LOAD_SYMBOL(lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double)
  LOAD_SYMBOL(lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_bias_tensor)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool)
  LOAD_SYMBOL(lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor)
  LOAD_SYMBOL(lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern_mm_tensor_tensor)
  LOAD_SYMBOL(lantern_mm_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__sparse_mm_tensor_tensor)
  LOAD_SYMBOL(lantern_mode_tensor_intt_bool)
  LOAD_SYMBOL(lantern_mode_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_mode_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_mode_out_tensor_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_mul_tensor_tensor)
  LOAD_SYMBOL(lantern_mul__tensor_tensor)
  LOAD_SYMBOL(lantern_mul_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_mul_tensor_scalar)
  LOAD_SYMBOL(lantern_mul__tensor_scalar)
  LOAD_SYMBOL(lantern_mv_tensor_tensor)
  LOAD_SYMBOL(lantern_mv_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_mvlgamma_tensor_intt)
  LOAD_SYMBOL(lantern_mvlgamma__tensor_intt)
  LOAD_SYMBOL(lantern_narrow_copy_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_narrow_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double)
  LOAD_SYMBOL(lantern_batch_norm_stats_tensor_double)
  LOAD_SYMBOL(lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double)
  LOAD_SYMBOL(lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt)
  LOAD_SYMBOL(lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_intarrayref)
  LOAD_SYMBOL(lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool)
  LOAD_SYMBOL(lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool)
  LOAD_SYMBOL(lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_batch_norm_update_stats_tensor_tensor_tensor_double)
  LOAD_SYMBOL(lantern__nnpack_available)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref)
  LOAD_SYMBOL(lantern_ones_intarrayref_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_ones_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_ones_out_tensor_intarrayref)
  LOAD_SYMBOL(lantern_ones_like_tensor)
  LOAD_SYMBOL(lantern_ones_like_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_pairwise_distance_tensor_tensor_double_double_bool)
  LOAD_SYMBOL(lantern_cdist_tensor_tensor_double)
  LOAD_SYMBOL(lantern__cdist_backward_tensor_tensor_tensor_double_tensor)
  LOAD_SYMBOL(lantern_pdist_tensor_double)
  LOAD_SYMBOL(lantern__pdist_forward_tensor_double)
  LOAD_SYMBOL(lantern__pdist_backward_tensor_tensor_double_tensor)
  LOAD_SYMBOL(lantern_cosine_similarity_tensor_tensor_intt_double)
  LOAD_SYMBOL(lantern_permute_tensor_intarrayref)
  LOAD_SYMBOL(lantern_numpy_t_tensor)
  LOAD_SYMBOL(lantern_pixel_shuffle_tensor_intt)
  LOAD_SYMBOL(lantern_is_pinned_tensor)
  LOAD_SYMBOL(lantern_pin_memory_tensor)
  LOAD_SYMBOL(lantern_pinverse_tensor_double)
  LOAD_SYMBOL(lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt)
  LOAD_SYMBOL(lantern_scalar_tensor_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_rand_intarrayref_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_rand_intarrayref_generator_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_rand_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_rand_intarrayref_generator_tensoroptions)
  LOAD_SYMBOL(lantern_rand_out_tensor_intarrayref)
  LOAD_SYMBOL(lantern_rand_out_tensor_intarrayref_generator)
  LOAD_SYMBOL(lantern_rand_like_tensor)
  LOAD_SYMBOL(lantern_rand_like_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_randint_intt_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_randint_intt_intarrayref_generator_tensoroptions)
  LOAD_SYMBOL(lantern_randint_intt_intt_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_randint_intt_intt_intarrayref_generator_tensoroptions)
  LOAD_SYMBOL(lantern_randint_out_tensor_intt_intarrayref)
  LOAD_SYMBOL(lantern_randint_out_tensor_intt_intarrayref_generator)
  LOAD_SYMBOL(lantern_randint_out_tensor_intt_intt_intarrayref)
  LOAD_SYMBOL(lantern_randint_out_tensor_intt_intt_intarrayref_generator)
  LOAD_SYMBOL(lantern_randint_like_tensor_intt)
  LOAD_SYMBOL(lantern_randint_like_tensor_intt_intt)
  LOAD_SYMBOL(lantern_randint_like_tensor_intt_tensoroptions)
  LOAD_SYMBOL(lantern_randint_like_tensor_intt_intt_tensoroptions)
  LOAD_SYMBOL(lantern_randn_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_randn_intarrayref_generator_tensoroptions)
  LOAD_SYMBOL(lantern_randn_intarrayref_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_randn_intarrayref_generator_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_randn_out_tensor_intarrayref)
  LOAD_SYMBOL(lantern_randn_out_tensor_intarrayref_generator)
  LOAD_SYMBOL(lantern_randn_like_tensor)
  LOAD_SYMBOL(lantern_randn_like_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_randperm_intt_tensoroptions)
  LOAD_SYMBOL(lantern_randperm_intt_generator_tensoroptions)
  LOAD_SYMBOL(lantern_randperm_out_tensor_intt)
  LOAD_SYMBOL(lantern_randperm_out_tensor_intt_generator)
  LOAD_SYMBOL(lantern_range_scalar_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_range_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_range_out_tensor_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_reciprocal_tensor)
  LOAD_SYMBOL(lantern_reciprocal__tensor)
  LOAD_SYMBOL(lantern_reciprocal_out_tensor_tensor)
  LOAD_SYMBOL(lantern_neg_tensor)
  LOAD_SYMBOL(lantern_neg__tensor)
  LOAD_SYMBOL(lantern_neg_out_tensor_tensor)
  LOAD_SYMBOL(lantern_repeat_tensor_intarrayref)
  LOAD_SYMBOL(lantern_repeat_interleave_tensor)
  LOAD_SYMBOL(lantern_repeat_interleave_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_repeat_interleave_tensor_intt_intt)
  LOAD_SYMBOL(lantern_reshape_tensor_intarrayref)
  LOAD_SYMBOL(lantern__mkldnn_reshape_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reshape_as_tensor_tensor)
  LOAD_SYMBOL(lantern_round_tensor)
  LOAD_SYMBOL(lantern_round__tensor)
  LOAD_SYMBOL(lantern_round_out_tensor_tensor)
  LOAD_SYMBOL(lantern_rrelu_tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_rrelu__tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_relu_tensor)
  LOAD_SYMBOL(lantern_relu__tensor)
  LOAD_SYMBOL(lantern_prelu_tensor_tensor)
  LOAD_SYMBOL(lantern_prelu_backward_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_gelu_tensor)
  LOAD_SYMBOL(lantern_gelu_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_hardshrink_tensor_scalar)
  LOAD_SYMBOL(lantern_hardshrink_backward_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_rsqrt_tensor)
  LOAD_SYMBOL(lantern_rsqrt__tensor)
  LOAD_SYMBOL(lantern_rsqrt_out_tensor_tensor)
  LOAD_SYMBOL(lantern_select_tensor_dimname_intt)
  LOAD_SYMBOL(lantern_select_tensor_intt_intt)
  LOAD_SYMBOL(lantern_selu_tensor)
  LOAD_SYMBOL(lantern_selu__tensor)
  LOAD_SYMBOL(lantern_celu_tensor_scalar)
  LOAD_SYMBOL(lantern_celu__tensor_scalar)
  LOAD_SYMBOL(lantern_sigmoid_tensor)
  LOAD_SYMBOL(lantern_sigmoid__tensor)
  LOAD_SYMBOL(lantern_sigmoid_out_tensor_tensor)
  LOAD_SYMBOL(lantern_sin_tensor)
  LOAD_SYMBOL(lantern_sin__tensor)
  LOAD_SYMBOL(lantern_sin_out_tensor_tensor)
  LOAD_SYMBOL(lantern_sinh_tensor)
  LOAD_SYMBOL(lantern_sinh__tensor)
  LOAD_SYMBOL(lantern_sinh_out_tensor_tensor)
  LOAD_SYMBOL(lantern_detach_tensor)
  LOAD_SYMBOL(lantern_detach__tensor)
  LOAD_SYMBOL(lantern_size_tensor_intt)
  LOAD_SYMBOL(lantern_size_tensor_dimname)
  LOAD_SYMBOL(lantern_slice_tensor_intt_intt_intt_intt)
  LOAD_SYMBOL(lantern_slogdet_tensor)
  LOAD_SYMBOL(lantern_smm_tensor_tensor)
  LOAD_SYMBOL(lantern_softmax_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_softmax_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern__softmax_tensor_intt_bool)
  LOAD_SYMBOL(lantern__softmax_backward_data_tensor_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_split_tensor_intt_intt)
  LOAD_SYMBOL(lantern_split_with_sizes_tensor_intarrayref_intt)
  LOAD_SYMBOL(lantern_squeeze_tensor)
  LOAD_SYMBOL(lantern_squeeze_tensor_intt)
  LOAD_SYMBOL(lantern_squeeze_tensor_dimname)
  LOAD_SYMBOL(lantern_squeeze__tensor)
  LOAD_SYMBOL(lantern_squeeze__tensor_intt)
  LOAD_SYMBOL(lantern_squeeze__tensor_dimname)
  LOAD_SYMBOL(lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_stack_tensorlist_intt)
  LOAD_SYMBOL(lantern_stack_out_tensor_tensorlist_intt)
  LOAD_SYMBOL(lantern_stft_tensor_intt_intt_intt_tensor_bool_bool)
  LOAD_SYMBOL(lantern_stride_tensor_intt)
  LOAD_SYMBOL(lantern_stride_tensor_dimname)
  LOAD_SYMBOL(lantern_sum_tensor_scalartype)
  LOAD_SYMBOL(lantern_sum_tensor_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_sum_tensor_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_sum_to_size_tensor_intarrayref)
  LOAD_SYMBOL(lantern_sqrt_tensor)
  LOAD_SYMBOL(lantern_sqrt__tensor)
  LOAD_SYMBOL(lantern_sqrt_out_tensor_tensor)
  LOAD_SYMBOL(lantern_std_tensor_bool)
  LOAD_SYMBOL(lantern_std_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_std_mean_tensor_bool)
  LOAD_SYMBOL(lantern_std_mean_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_std_mean_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_std_out_tensor_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_std_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_std_out_tensor_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_prod_tensor_scalartype)
  LOAD_SYMBOL(lantern_prod_tensor_intt_bool_scalartype)
  LOAD_SYMBOL(lantern_prod_out_tensor_tensor_intt_bool_scalartype)
  LOAD_SYMBOL(lantern_prod_tensor_dimname_bool_scalartype)
  LOAD_SYMBOL(lantern_prod_out_tensor_tensor_dimname_bool_scalartype)
  LOAD_SYMBOL(lantern_t_tensor)
  LOAD_SYMBOL(lantern_t__tensor)
  LOAD_SYMBOL(lantern_tan_tensor)
  LOAD_SYMBOL(lantern_tan__tensor)
  LOAD_SYMBOL(lantern_tan_out_tensor_tensor)
  LOAD_SYMBOL(lantern_tanh_tensor)
  LOAD_SYMBOL(lantern_tanh__tensor)
  LOAD_SYMBOL(lantern_tanh_out_tensor_tensor)
  LOAD_SYMBOL(lantern_tensordot_tensor_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_threshold_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold_out_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold_backward_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_transpose_tensor_intt_intt)
  LOAD_SYMBOL(lantern_transpose_tensor_dimname_dimname)
  LOAD_SYMBOL(lantern__mkldnn_transpose_tensor_intt_intt)
  LOAD_SYMBOL(lantern_transpose__tensor_intt_intt)
  LOAD_SYMBOL(lantern__mkldnn_transpose__tensor_intt_intt)
  LOAD_SYMBOL(lantern_one_hot_tensor_intt)
  LOAD_SYMBOL(lantern_flip_tensor_intarrayref)
  LOAD_SYMBOL(lantern_roll_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_rot90_tensor_intt_intarrayref)
  LOAD_SYMBOL(lantern_trapz_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_trapz_tensor_double_intt)
  LOAD_SYMBOL(lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt)
  LOAD_SYMBOL(lantern_trunc_tensor)
  LOAD_SYMBOL(lantern_trunc__tensor)
  LOAD_SYMBOL(lantern_trunc_out_tensor_tensor)
  LOAD_SYMBOL(lantern_type_as_tensor_tensor)
  LOAD_SYMBOL(lantern__has_compatible_shallow_copy_type_tensor_tensor)
  LOAD_SYMBOL(lantern__unique_tensor_bool_bool)
  LOAD_SYMBOL(lantern_unique_dim_tensor_intt_bool_bool_bool)
  LOAD_SYMBOL(lantern_unique_consecutive_tensor_bool_bool_intt)
  LOAD_SYMBOL(lantern_unique_dim_consecutive_tensor_intt_bool_bool)
  LOAD_SYMBOL(lantern__unique2_tensor_bool_bool_bool)
  LOAD_SYMBOL(lantern__unsafe_view_tensor_intarrayref)
  LOAD_SYMBOL(lantern_unsqueeze_tensor_intt)
  LOAD_SYMBOL(lantern_unsqueeze__tensor_intt)
  LOAD_SYMBOL(lantern_var_tensor_bool)
  LOAD_SYMBOL(lantern_var_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_var_out_tensor_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_var_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_var_out_tensor_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_var_mean_tensor_bool)
  LOAD_SYMBOL(lantern_var_mean_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_var_mean_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_view_as_tensor_tensor)
  LOAD_SYMBOL(lantern_where_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_where_tensor)
  LOAD_SYMBOL(lantern__s_where_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_norm_except_dim_tensor_intt_intt)
  LOAD_SYMBOL(lantern__weight_norm_tensor_tensor_intt)
  LOAD_SYMBOL(lantern__weight_norm_cuda_interface_tensor_tensor_intt)
  LOAD_SYMBOL(lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_zeros_intarrayref_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_zeros_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_zeros_out_tensor_intarrayref)
  LOAD_SYMBOL(lantern_zeros_like_tensor)
  LOAD_SYMBOL(lantern_zeros_like_tensor_tensoroptions)
  LOAD_SYMBOL(lantern__standard_gamma_grad_tensor_tensor)
  LOAD_SYMBOL(lantern__standard_gamma_tensor_generator)
  LOAD_SYMBOL(lantern__dirichlet_grad_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__sample_dirichlet_tensor_generator)
  LOAD_SYMBOL(lantern_poisson_tensor_generator)
  LOAD_SYMBOL(lantern_native_norm_tensor_scalar)
  LOAD_SYMBOL(lantern__sparse_sum_tensor)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_scalartype)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_intarrayref)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_intarrayref_scalartype)
  LOAD_SYMBOL(lantern__sparse_sum_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_norm_tensor_scalar_scalartype)
  LOAD_SYMBOL(lantern_norm_tensor_scalar)
  LOAD_SYMBOL(lantern_norm_tensor_scalar_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_tensor_scalar_intarrayref_bool)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_intarrayref_bool)
  LOAD_SYMBOL(lantern_norm_tensor_scalar_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_tensor_scalar_dimnamelist_bool)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool)
  LOAD_SYMBOL(lantern_frobenius_norm_tensor)
  LOAD_SYMBOL(lantern_frobenius_norm_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_tensor_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_out_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_clone_tensor)
  LOAD_SYMBOL(lantern_resize_as__tensor_tensor)
  LOAD_SYMBOL(lantern_pow_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_pow_tensor_scalar)
  LOAD_SYMBOL(lantern_zero__tensor)
  LOAD_SYMBOL(lantern_sub_out_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_sub_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_sub__tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_sub_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_sub__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_rsub_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_rsub_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addmm__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_tensor_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_sparse_resize__tensor_intarrayref_intt_intt)
  LOAD_SYMBOL(lantern_sparse_resize_and_clear__tensor_intarrayref_intt_intt)
  LOAD_SYMBOL(lantern_sparse_mask_tensor_tensor)
  LOAD_SYMBOL(lantern_to_dense_tensor)
  LOAD_SYMBOL(lantern_to_dense_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_sparse_dim_tensor)
  LOAD_SYMBOL(lantern__dimi_tensor)
  LOAD_SYMBOL(lantern_dense_dim_tensor)
  LOAD_SYMBOL(lantern__dimv_tensor)
  LOAD_SYMBOL(lantern__nnz_tensor)
  LOAD_SYMBOL(lantern_coalesce_tensor)
  LOAD_SYMBOL(lantern_is_coalesced_tensor)
  LOAD_SYMBOL(lantern__indices_tensor)
  LOAD_SYMBOL(lantern__values_tensor)
  LOAD_SYMBOL(lantern__coalesced__tensor_bool)
  LOAD_SYMBOL(lantern_indices_tensor)
  LOAD_SYMBOL(lantern_values_tensor)
  LOAD_SYMBOL(lantern_hspmm_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_hspmm_tensor_tensor)
  LOAD_SYMBOL(lantern_copy_sparse_to_sparse__tensor_tensor_bool)
  LOAD_SYMBOL(lantern_numel_tensor)
  LOAD_SYMBOL(lantern_unbind_tensor_intt)
  LOAD_SYMBOL(lantern_unbind_tensor_dimname)
  LOAD_SYMBOL(lantern_to_sparse_tensor_intt)
  LOAD_SYMBOL(lantern_to_sparse_tensor)
  LOAD_SYMBOL(lantern_to_mkldnn_tensor)
  LOAD_SYMBOL(lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_to_mkldnn_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_quantize_per_tensor_tensor_double_intt_scalartype)
  LOAD_SYMBOL(lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_dequantize_tensor)
  LOAD_SYMBOL(lantern_q_scale_tensor)
  LOAD_SYMBOL(lantern_q_zero_point_tensor)
  LOAD_SYMBOL(lantern_q_per_channel_scales_tensor)
  LOAD_SYMBOL(lantern_q_per_channel_zero_points_tensor)
  LOAD_SYMBOL(lantern_q_per_channel_axis_tensor)
  LOAD_SYMBOL(lantern_int_repr_tensor)
  LOAD_SYMBOL(lantern__make_per_tensor_quantized_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_qscheme_tensor)
  LOAD_SYMBOL(lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt)
  LOAD_SYMBOL(lantern_fake_quantize_per_tensor_affine_backward_tensor_tensor_double_intt_intt_intt)
  LOAD_SYMBOL(lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_fake_quantize_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_to_tensor_tensoroptions_bool_bool)
  LOAD_SYMBOL(lantern_to_tensor_device_scalartype_bool_bool)
  LOAD_SYMBOL(lantern_to_tensor_scalartype_bool_bool)
  LOAD_SYMBOL(lantern_to_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern_meshgrid_tensorlist)
  LOAD_SYMBOL(lantern_cartesian_prod_tensorlist)
  LOAD_SYMBOL(lantern_combinations_tensor_intt_bool)
  LOAD_SYMBOL(lantern_item_tensor)
  LOAD_SYMBOL(lantern_result_type_tensor_tensor)
  LOAD_SYMBOL(lantern_result_type_tensor_scalar)
  LOAD_SYMBOL(lantern_result_type_scalar_tensor)
  LOAD_SYMBOL(lantern_result_type_scalar_scalar)
  LOAD_SYMBOL(lantern_can_cast_scalartype_scalartype)
  LOAD_SYMBOL(lantern_promote_types_scalartype_scalartype)
  LOAD_SYMBOL(lantern__local_scalar_dense_tensor)
  LOAD_SYMBOL(lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool)
  LOAD_SYMBOL(lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool)
  LOAD_SYMBOL(lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool)
  LOAD_SYMBOL(lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)
  LOAD_SYMBOL(lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)
  LOAD_SYMBOL(lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)
  LOAD_SYMBOL(lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)
  LOAD_SYMBOL(lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)
  LOAD_SYMBOL(lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)
  LOAD_SYMBOL(lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_quantized_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool_scalartype_bool)
  LOAD_SYMBOL(lantern_quantized_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool)
  LOAD_SYMBOL(lantern_quantized_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool)
  LOAD_SYMBOL(lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern__pack_padded_sequence_tensor_tensor_bool)
  LOAD_SYMBOL(lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool)
  LOAD_SYMBOL(lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt)
  LOAD_SYMBOL(lantern_set__tensor_storage)
  LOAD_SYMBOL(lantern_set__tensor_storage_intt_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_set__tensor_tensor)
  LOAD_SYMBOL(lantern_set__tensor)
  LOAD_SYMBOL(lantern_set_quantizer__tensor_constquantizerptr)
  LOAD_SYMBOL(lantern_is_set_to_tensor_tensor)
  LOAD_SYMBOL(lantern_masked_fill__tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_masked_fill_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_masked_fill__tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_masked_fill_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_masked_scatter__tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_masked_scatter_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_view_tensor_intarrayref)
  LOAD_SYMBOL(lantern_put__tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_index_add__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_add_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_add_tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_index_fill__tensor_intt_tensor_scalar)
  LOAD_SYMBOL(lantern_index_fill_tensor_intt_tensor_scalar)
  LOAD_SYMBOL(lantern_index_fill__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_fill_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_index_fill__tensor_dimname_tensor_scalar)
  LOAD_SYMBOL(lantern_index_fill__tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_index_fill_tensor_dimname_tensor_scalar)
  LOAD_SYMBOL(lantern_index_fill_tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter__tensor_intt_tensor_scalar)
  LOAD_SYMBOL(lantern_scatter_tensor_intt_tensor_scalar)
  LOAD_SYMBOL(lantern_scatter_tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter_tensor_dimname_tensor_scalar)
  LOAD_SYMBOL(lantern_scatter_add__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter_add_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_scatter_add_tensor_dimname_tensor_tensor)
  LOAD_SYMBOL(lantern_lt__tensor_scalar)
  LOAD_SYMBOL(lantern_lt__tensor_tensor)
  LOAD_SYMBOL(lantern_gt__tensor_scalar)
  LOAD_SYMBOL(lantern_gt__tensor_tensor)
  LOAD_SYMBOL(lantern_le__tensor_scalar)
  LOAD_SYMBOL(lantern_le__tensor_tensor)
  LOAD_SYMBOL(lantern_ge__tensor_scalar)
  LOAD_SYMBOL(lantern_ge__tensor_tensor)
  LOAD_SYMBOL(lantern_eq__tensor_scalar)
  LOAD_SYMBOL(lantern_eq__tensor_tensor)
  LOAD_SYMBOL(lantern_ne__tensor_scalar)
  LOAD_SYMBOL(lantern_ne__tensor_tensor)
  LOAD_SYMBOL(lantern___and___tensor_scalar)
  LOAD_SYMBOL(lantern___and___tensor_tensor)
  LOAD_SYMBOL(lantern___iand___tensor_scalar)
  LOAD_SYMBOL(lantern___iand___tensor_tensor)
  LOAD_SYMBOL(lantern___or___tensor_scalar)
  LOAD_SYMBOL(lantern___or___tensor_tensor)
  LOAD_SYMBOL(lantern___ior___tensor_scalar)
  LOAD_SYMBOL(lantern___ior___tensor_tensor)
  LOAD_SYMBOL(lantern___xor___tensor_scalar)
  LOAD_SYMBOL(lantern___xor___tensor_tensor)
  LOAD_SYMBOL(lantern___ixor___tensor_scalar)
  LOAD_SYMBOL(lantern___ixor___tensor_tensor)
  LOAD_SYMBOL(lantern___lshift___tensor_scalar)
  LOAD_SYMBOL(lantern___lshift___tensor_tensor)
  LOAD_SYMBOL(lantern___ilshift___tensor_scalar)
  LOAD_SYMBOL(lantern___ilshift___tensor_tensor)
  LOAD_SYMBOL(lantern___rshift___tensor_scalar)
  LOAD_SYMBOL(lantern___rshift___tensor_tensor)
  LOAD_SYMBOL(lantern___irshift___tensor_scalar)
  LOAD_SYMBOL(lantern___irshift___tensor_tensor)
  LOAD_SYMBOL(lantern_lgamma__tensor)
  LOAD_SYMBOL(lantern_atan2__tensor_tensor)
  LOAD_SYMBOL(lantern_tril__tensor_intt)
  LOAD_SYMBOL(lantern_triu__tensor_intt)
  LOAD_SYMBOL(lantern_digamma__tensor)
  LOAD_SYMBOL(lantern_polygamma__tensor_intt)
  LOAD_SYMBOL(lantern_renorm__tensor_scalar_intt_scalar)
  LOAD_SYMBOL(lantern_pow__tensor_scalar)
  LOAD_SYMBOL(lantern_pow__tensor_tensor)
  LOAD_SYMBOL(lantern_lerp__tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_lerp__tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fmod__tensor_scalar)
  LOAD_SYMBOL(lantern_fmod__tensor_tensor)
  LOAD_SYMBOL(lantern_remainder__tensor_scalar)
  LOAD_SYMBOL(lantern_remainder__tensor_tensor)
  LOAD_SYMBOL(lantern_addbmm__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addbmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_addcdiv__tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_random__tensor_intt_intt_generator)
  LOAD_SYMBOL(lantern_random__tensor_intt_generator)
  LOAD_SYMBOL(lantern_random__tensor_generator)
  LOAD_SYMBOL(lantern_uniform__tensor_double_double_generator)
  LOAD_SYMBOL(lantern_normal__tensor_double_double_generator)
  LOAD_SYMBOL(lantern_cauchy__tensor_double_double_generator)
  LOAD_SYMBOL(lantern_log_normal__tensor_double_double_generator)
  LOAD_SYMBOL(lantern_exponential__tensor_double_generator)
  LOAD_SYMBOL(lantern_geometric__tensor_double_generator)
  LOAD_SYMBOL(lantern_diag_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_diag_tensor_intt)
  LOAD_SYMBOL(lantern_cross_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_cross_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_triu_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_triu_tensor_intt)
  LOAD_SYMBOL(lantern_tril_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_tril_tensor_intt)
  LOAD_SYMBOL(lantern_tril_indices_intt_intt_intt_tensoroptions)
  LOAD_SYMBOL(lantern_triu_indices_intt_intt_intt_tensoroptions)
  LOAD_SYMBOL(lantern_trace_tensor)
  LOAD_SYMBOL(lantern_ne_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_ne_tensor_scalar)
  LOAD_SYMBOL(lantern_ne_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_ne_tensor_tensor)
  LOAD_SYMBOL(lantern_eq_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_eq_tensor_scalar)
  LOAD_SYMBOL(lantern_eq_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_eq_tensor_tensor)
  LOAD_SYMBOL(lantern_ge_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_ge_tensor_scalar)
  LOAD_SYMBOL(lantern_ge_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_ge_tensor_tensor)
  LOAD_SYMBOL(lantern_le_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_le_tensor_scalar)
  LOAD_SYMBOL(lantern_le_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_le_tensor_tensor)
  LOAD_SYMBOL(lantern_gt_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_gt_tensor_scalar)
  LOAD_SYMBOL(lantern_gt_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_gt_tensor_tensor)
  LOAD_SYMBOL(lantern_lt_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_lt_tensor_scalar)
  LOAD_SYMBOL(lantern_lt_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_lt_tensor_tensor)
  LOAD_SYMBOL(lantern_take_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_take_tensor_tensor)
  LOAD_SYMBOL(lantern_index_select_out_tensor_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_index_select_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_index_select_out_tensor_tensor_dimname_tensor)
  LOAD_SYMBOL(lantern_index_select_tensor_dimname_tensor)
  LOAD_SYMBOL(lantern_masked_select_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_masked_select_tensor_tensor)
  LOAD_SYMBOL(lantern_nonzero_out_tensor_tensor)
  LOAD_SYMBOL(lantern_nonzero_tensor)
  LOAD_SYMBOL(lantern_nonzero_numpy_tensor)
  LOAD_SYMBOL(lantern_gather_out_tensor_tensor_intt_tensor_bool)
  LOAD_SYMBOL(lantern_gather_tensor_intt_tensor_bool)
  LOAD_SYMBOL(lantern_gather_out_tensor_tensor_dimname_tensor_bool)
  LOAD_SYMBOL(lantern_gather_tensor_dimname_tensor_bool)
  LOAD_SYMBOL(lantern__gather_sparse_backward_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_addcmul_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_addcmul__tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_addcdiv_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_lstsq_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_lstsq_tensor_tensor)
  LOAD_SYMBOL(lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool)
  LOAD_SYMBOL(lantern_triangular_solve_tensor_tensor_bool_bool_bool)
  LOAD_SYMBOL(lantern__triangular_solve_helper_tensor_tensor_bool_bool_bool)
  LOAD_SYMBOL(lantern_symeig_out_tensor_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern_symeig_tensor_bool_bool)
  LOAD_SYMBOL(lantern__symeig_helper_tensor_bool_bool)
  LOAD_SYMBOL(lantern_eig_out_tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_eig_tensor_bool)
  LOAD_SYMBOL(lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern_svd_tensor_bool_bool)
  LOAD_SYMBOL(lantern__svd_helper_tensor_bool_bool)
  LOAD_SYMBOL(lantern_cholesky_out_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_cholesky_tensor_bool)
  LOAD_SYMBOL(lantern__cholesky_helper_tensor_bool)
  LOAD_SYMBOL(lantern_cholesky_solve_out_tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_cholesky_solve_tensor_tensor_bool)
  LOAD_SYMBOL(lantern__cholesky_solve_helper_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_solve_tensor_tensor)
  LOAD_SYMBOL(lantern_solve_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__solve_helper_tensor_tensor)
  LOAD_SYMBOL(lantern_cholesky_inverse_out_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_cholesky_inverse_tensor_bool)
  LOAD_SYMBOL(lantern_qr_out_tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_qr_tensor_bool)
  LOAD_SYMBOL(lantern__qr_helper_tensor_bool)
  LOAD_SYMBOL(lantern_geqrf_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_geqrf_tensor)
  LOAD_SYMBOL(lantern_orgqr_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_orgqr_tensor_tensor)
  LOAD_SYMBOL(lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern_ormqr_tensor_tensor_tensor_bool_bool)
  LOAD_SYMBOL(lantern__lu_with_info_tensor_bool_bool)
  LOAD_SYMBOL(lantern_lu_solve_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_lu_solve_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern__lu_solve_helper_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_multinomial_out_tensor_tensor_intt_bool_generator)
  LOAD_SYMBOL(lantern_multinomial_tensor_intt_bool_generator)
  LOAD_SYMBOL(lantern__multinomial_alias_setup_tensor)
  LOAD_SYMBOL(lantern__multinomial_alias_draw_tensor_tensor_intt_generator)
  LOAD_SYMBOL(lantern_lgamma_out_tensor_tensor)
  LOAD_SYMBOL(lantern_lgamma_tensor)
  LOAD_SYMBOL(lantern_digamma_out_tensor_tensor)
  LOAD_SYMBOL(lantern_digamma_tensor)
  LOAD_SYMBOL(lantern_polygamma_out_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_polygamma_intt_tensor)
  LOAD_SYMBOL(lantern_erfinv_tensor)
  LOAD_SYMBOL(lantern_erfinv__tensor)
  LOAD_SYMBOL(lantern_erfinv_out_tensor_tensor)
  LOAD_SYMBOL(lantern_sign_tensor)
  LOAD_SYMBOL(lantern_sign__tensor)
  LOAD_SYMBOL(lantern_sign_out_tensor_tensor)
  LOAD_SYMBOL(lantern_dist_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_atan2_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_atan2_tensor_tensor)
  LOAD_SYMBOL(lantern_lerp_out_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_lerp_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_lerp_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_lerp_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_histc_out_tensor_tensor_intt_scalar_scalar)
  LOAD_SYMBOL(lantern_histc_tensor_intt_scalar_scalar)
  LOAD_SYMBOL(lantern_fmod_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_fmod_tensor_scalar)
  LOAD_SYMBOL(lantern_fmod_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fmod_tensor_tensor)
  LOAD_SYMBOL(lantern_remainder_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_remainder_tensor_scalar)
  LOAD_SYMBOL(lantern_remainder_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_remainder_tensor_tensor)
  LOAD_SYMBOL(lantern_min_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_min_tensor_tensor)
  LOAD_SYMBOL(lantern_min_tensor)
  LOAD_SYMBOL(lantern_max_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_max_tensor_tensor)
  LOAD_SYMBOL(lantern_max_tensor)
  LOAD_SYMBOL(lantern_median_tensor)
  LOAD_SYMBOL(lantern_sort_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_sort_tensor_intt_bool)
  LOAD_SYMBOL(lantern_sort_out_tensor_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_sort_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_argsort_tensor_intt_bool)
  LOAD_SYMBOL(lantern_argsort_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool)
  LOAD_SYMBOL(lantern_topk_tensor_intt_intt_bool_bool)
  LOAD_SYMBOL(lantern_all_tensor)
  LOAD_SYMBOL(lantern_any_tensor)
  LOAD_SYMBOL(lantern_renorm_out_tensor_tensor_scalar_intt_scalar)
  LOAD_SYMBOL(lantern_renorm_tensor_scalar_intt_scalar)
  LOAD_SYMBOL(lantern_unfold_tensor_intt_intt_intt)
  LOAD_SYMBOL(lantern_equal_tensor_tensor)
  LOAD_SYMBOL(lantern_pow_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_pow_tensor_tensor)
  LOAD_SYMBOL(lantern_pow_out_tensor_scalar_tensor)
  LOAD_SYMBOL(lantern_pow_scalar_tensor)
  LOAD_SYMBOL(lantern_normal_out_tensor_tensor_double_generator)
  LOAD_SYMBOL(lantern_normal_out_tensor_double_tensor_generator)
  LOAD_SYMBOL(lantern_normal_out_tensor_tensor_tensor_generator)
  LOAD_SYMBOL(lantern_normal_out_tensor_double_double_intarrayref_generator)
  LOAD_SYMBOL(lantern_alias_tensor)
  LOAD_SYMBOL(lantern__addr_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__addr__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__addr_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__index_copy__tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern__cumsum_tensor_intt)
  LOAD_SYMBOL(lantern__cumsum_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern__cumprod_tensor_intt)
  LOAD_SYMBOL(lantern__cumprod_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern__var_tensor_bool)
  LOAD_SYMBOL(lantern__std_tensor_bool)
  LOAD_SYMBOL(lantern__cat_tensorlist_intt)
  LOAD_SYMBOL(lantern__cat_out_tensor_tensorlist_intt)
  LOAD_SYMBOL(lantern__mode_tensor_intt_bool)
  LOAD_SYMBOL(lantern__mode_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern__max_tensor_intt_bool)
  LOAD_SYMBOL(lantern__max_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern__min_tensor_intt_bool)
  LOAD_SYMBOL(lantern__min_out_tensor_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_binary_cross_entropy_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_mse_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_mse_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_mse_loss_backward_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_l1_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_l1_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_l1_loss_backward_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt)
  LOAD_SYMBOL(lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt)
  LOAD_SYMBOL(lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt)
  LOAD_SYMBOL(lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_forward_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor)
  LOAD_SYMBOL(lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor)
  LOAD_SYMBOL(lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor)
  LOAD_SYMBOL(lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor)
  LOAD_SYMBOL(lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_smooth_l1_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_soft_margin_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_soft_margin_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_elu_out_tensor_tensor_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_elu_tensor_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_elu_backward_tensor_scalar_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_elu__tensor_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_glu_out_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_glu_tensor_intt)
  LOAD_SYMBOL(lantern_glu_backward_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_glu_backward_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_hardtanh_out_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_hardtanh_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_hardtanh_backward_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_hardtanh__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_leaky_relu_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_leaky_relu_tensor_scalar)
  LOAD_SYMBOL(lantern_leaky_relu_backward_out_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_leaky_relu_backward_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_leaky_relu__tensor_scalar)
  LOAD_SYMBOL(lantern_log_sigmoid_out_tensor_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_forward_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_forward_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_backward_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_rrelu_with_noise_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_bool)
  LOAD_SYMBOL(lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool)
  LOAD_SYMBOL(lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_softplus_out_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_softplus_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_softshrink_out_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_softshrink_tensor_scalar)
  LOAD_SYMBOL(lantern_softshrink_backward_out_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_softshrink_backward_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_avg_pool2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern__adaptive_avg_pool2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern__adaptive_avg_pool2d_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor)
  LOAD_SYMBOL(lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool2d_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad1d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad1d_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad2d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_reflection_pad2d_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad1d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad1d_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad2d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad2d_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad3d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad3d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_replication_pad3d_backward_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_linear1d_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_upsample_nearest1d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest2d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest3d_out_tensor_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest3d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_sigmoid_backward_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_sigmoid_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_tanh_backward_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_tanh_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor)
  LOAD_SYMBOL(lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)
  LOAD_SYMBOL(lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor)
  LOAD_SYMBOL(lantern_thnn_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool)
  LOAD_SYMBOL(lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)
  LOAD_SYMBOL(lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool)
  LOAD_SYMBOL(lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref)
  /* Autogen Symbols -- End */
  
  return true;
}

#endif
#endif
