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
#ifdef _WIN32
#define LANTERN_API extern "C" __declspec(dllexport)
#endif
#else
#define LANTERN_PTR *
#endif

#ifndef LANTERN_API
#define LANTERN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
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
LANTERN_API void* (LANTERN_PTR lantern_align_tensors_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern__debug_has_internal_overlap_tensor)(void* self);
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
LANTERN_API void* (LANTERN_PTR lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool)(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool1d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar)(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_affine_grid_generator_tensor_intarrayref_bool)(void* theta, void* size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_affine_grid_generator_backward_tensor_intarrayref_bool)(void* grad, void* size, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_all_out_tensor_tensor_intt_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_all_out_tensor_tensor_dimname_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_any_out_tensor_tensor_intt_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_any_out_tensor_tensor_dimname_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_tensoroptions)(void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_scalar_tensoroptions)(void* start, void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_scalar_scalar_scalar_tensoroptions)(void* start, void* end, void* step, void* options);
LANTERN_API void* (LANTERN_PTR lantern_arange_out_tensor_scalar)(void* out, void* end);
LANTERN_API void* (LANTERN_PTR lantern__dim_arange_tensor_intt)(void* like, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar)(void* self, void* batch1, void* batch2, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_bartlett_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_bartlett_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern_bernoulli_out_tensor_tensor_generator)(void* out, void* self, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_bilinear_tensor_tensor_tensor_tensor)(void* input1, void* input2, void* weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt)(void* self, void* target, void* weight, void* pos_weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt)(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_blackman_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_blackman_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_broadcast_tensors_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_cat_tensorlist_intt)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_out_tensor_tensorlist_intt)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_tensorlist_dimname)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_cat_out_tensor_tensorlist_dimname)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_chain_matmul_tensorlist)(void* matrices);
LANTERN_API void* (LANTERN_PTR lantern_cudnn_is_acceptable_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_constant_pad_nd_tensor_intarrayref_scalar)(void* self, void* pad, void* value);
LANTERN_API void* (LANTERN_PTR lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups);
LANTERN_API void* (LANTERN_PTR lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding);
LANTERN_API void* (LANTERN_PTR lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_conv_tbc_tensor_tensor_tensor_intt)(void* self, void* weight, void* bias, void* pad);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt)(void* input1, void* input2, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_out_tensor_tensor_intt_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumsum_out_tensor_tensor_dimname_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_out_tensor_tensor_intt_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_cumprod_out_tensor_tensor_dimname_scalartype)(void* out, void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool)(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity);
LANTERN_API void* (LANTERN_PTR lantern_dot_out_tensor_tensor_tensor)(void* out, void* self, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_einsum_stdstring_tensorlist)(void* equation, void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_embedding_tensor_tensor_intt_bool_bool)(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse);
LANTERN_API void* (LANTERN_PTR lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool)(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse);
LANTERN_API void* (LANTERN_PTR lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool)(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor)(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights);
LANTERN_API void* (LANTERN_PTR lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat)(void* size, void* names, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_empty_out_tensor_intarrayref_memoryformat)(void* out, void* size, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_empty_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_empty_like_tensor_tensoroptions_memoryformat)(void* self, void* options, void* memory_format);
LANTERN_API void* (LANTERN_PTR lantern_eye_intt_tensoroptions)(void* n, void* options);
LANTERN_API void* (LANTERN_PTR lantern_eye_intt_intt_tensoroptions)(void* n, void* m, void* options);
LANTERN_API void* (LANTERN_PTR lantern_full_intarrayref_scalar_dimnamelist_tensoroptions)(void* size, void* fill_value, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_full_intarrayref_scalar_tensoroptions)(void* size, void* fill_value, void* options);
LANTERN_API void* (LANTERN_PTR lantern_full_out_tensor_intarrayref_scalar)(void* out, void* size, void* fill_value);
LANTERN_API void* (LANTERN_PTR lantern_full_like_tensor_scalar)(void* self, void* fill_value);
LANTERN_API void* (LANTERN_PTR lantern_full_like_tensor_scalar_tensoroptions)(void* self, void* fill_value, void* options);
LANTERN_API void* (LANTERN_PTR lantern_grid_sampler_tensor_tensor_intt_intt_bool)(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners);
LANTERN_API void* (LANTERN_PTR lantern_hann_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hann_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_tensoroptions)(void* window_length, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_tensoroptions)(void* window_length, void* periodic, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_double_tensoroptions)(void* window_length, void* periodic, void* alpha, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hamming_window_intt_bool_double_double_tensoroptions)(void* window_length, void* periodic, void* alpha, void* beta, void* options);
LANTERN_API void* (LANTERN_PTR lantern_hinge_embedding_loss_tensor_tensor_double_intt)(void* self, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_group_norm_tensor_intt_tensor_tensor_double_bool)(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern__cufft_get_plan_cache_size_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern__cufft_get_plan_cache_max_size_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern__cufft_set_plan_cache_max_size_intt_intt)(void* device_index, void* max_size);
LANTERN_API void* (LANTERN_PTR lantern__cufft_clear_plan_cache_intt)(void* device_index);
LANTERN_API void* (LANTERN_PTR lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool)(void* self, void* indices, void* values, void* accumulate, void* unsafe);
LANTERN_API void* (LANTERN_PTR lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled);
LANTERN_API void* (LANTERN_PTR lantern_inverse_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_isnan_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_kl_div_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool)(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable);
LANTERN_API void* (LANTERN_PTR lantern_linear_tensor_tensor_tensor)(void* input, void* weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor)(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor)(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_gemm_matrix_fp16_tensor)(void* input);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor)(void* input, void* packed_weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor)(void* input, void* packed_weight, void* bias);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix_tensor)(void* input);
LANTERN_API void* (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt)(void* input, void* K, void* N);
LANTERN_API void* (LANTERN_PTR lantern_linspace_scalar_scalar_intt_tensoroptions)(void* start, void* end, void* steps, void* options);
LANTERN_API void* (LANTERN_PTR lantern_logspace_scalar_scalar_intt_double_tensoroptions)(void* start, void* end, void* steps, void* base, void* options);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_logsumexp_out_tensor_tensor_dimnamelist_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt)(void* input1, void* input2, void* target, void* margin, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_matmul_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_matrix_rank_tensor_double_bool)(void* self, void* tol, void* symmetric);
LANTERN_API void* (LANTERN_PTR lantern_matrix_rank_tensor_bool)(void* self, void* symmetric);
LANTERN_API void* (LANTERN_PTR lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups);
LANTERN_API void* (LANTERN_PTR lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined);
LANTERN_API void* (LANTERN_PTR lantern__sparse_mm_tensor_tensor)(void* sparse, void* dense);
LANTERN_API void* (LANTERN_PTR lantern__nnpack_available)();
LANTERN_API void* (LANTERN_PTR lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref)(void* input, void* weight, void* bias, void* padding);
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
LANTERN_API void* (LANTERN_PTR lantern_pixel_shuffle_tensor_intt)(void* self, void* upscale_factor);
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
LANTERN_API void* (LANTERN_PTR lantern_range_scalar_scalar_scalar_tensoroptions)(void* start, void* end, void* step, void* options);
LANTERN_API void* (LANTERN_PTR lantern_range_scalar_scalar_tensoroptions)(void* start, void* end, void* options);
LANTERN_API void* (LANTERN_PTR lantern_rrelu_tensor_scalar_scalar_bool_generator)(void* self, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_rrelu__tensor_scalar_scalar_bool_generator)(void* self, void* lower, void* upper, void* training, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_selu_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_selu__tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_celu_tensor_scalar)(void* self, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_celu__tensor_scalar)(void* self, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_stack_tensorlist_intt)(void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_stack_out_tensor_tensorlist_intt)(void* out, void* tensors, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_std_out_tensor_tensor_intarrayref_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_std_out_tensor_tensor_dimnamelist_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_prod_out_tensor_tensor_intt_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_prod_out_tensor_tensor_dimname_bool_scalartype)(void* out, void* self, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_tensordot_tensor_tensor_intarrayref_intarrayref)(void* self, void* other, void* dims_self, void* dims_other);
LANTERN_API void* (LANTERN_PTR lantern_threshold_tensor_scalar_scalar)(void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold__tensor_scalar_scalar)(void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold_out_tensor_tensor_scalar_scalar)(void* out, void* self, void* threshold, void* value);
LANTERN_API void* (LANTERN_PTR lantern_threshold_backward_tensor_tensor_scalar)(void* grad_output, void* self, void* threshold);
LANTERN_API void* (LANTERN_PTR lantern_one_hot_tensor_intt)(void* self, void* num_classes);
LANTERN_API void* (LANTERN_PTR lantern_trapz_tensor_tensor_intt)(void* y, void* x, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_trapz_tensor_double_intt)(void* y, void* dx, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt)(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim);
LANTERN_API void* (LANTERN_PTR lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt)(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern__has_compatible_shallow_copy_type_tensor_tensor)(void* self, void* from);
LANTERN_API void* (LANTERN_PTR lantern__unsafe_view_tensor_intarrayref)(void* self, void* size);
LANTERN_API void* (LANTERN_PTR lantern_var_out_tensor_tensor_intarrayref_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_var_out_tensor_tensor_dimnamelist_bool_bool)(void* out, void* self, void* dim, void* unbiased, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_where_tensor)(void* condition);
LANTERN_API void* (LANTERN_PTR lantern_norm_except_dim_tensor_intt_intt)(void* v, void* pow, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__weight_norm_tensor_tensor_intt)(void* v, void* g, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_zeros_intarrayref_dimnamelist_tensoroptions)(void* size, void* names, void* options);
LANTERN_API void* (LANTERN_PTR lantern_zeros_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_zeros_out_tensor_intarrayref)(void* out, void* size);
LANTERN_API void* (LANTERN_PTR lantern_zeros_like_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_zeros_like_tensor_tensoroptions)(void* self, void* options);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_scalartype)(void* self, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_intarrayref)(void* self, void* dim);
LANTERN_API void* (LANTERN_PTR lantern__sparse_sum_tensor_intarrayref_scalartype)(void* self, void* dim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype)(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_intarrayref_bool)(void* out, void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype)(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype);
LANTERN_API void* (LANTERN_PTR lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool)(void* out, void* self, void* p, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_tensor_bool)(void* self, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_out_tensor_tensor_bool)(void* out, void* self, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_tensor_intarrayref_bool)(void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool)(void* out, void* self, void* dim, void* keepdim);
LANTERN_API void* (LANTERN_PTR lantern_rsub_tensor_tensor_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_rsub_tensor_scalar_scalar)(void* self, void* other, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar)(void* self, void* sparse, void* dense, void* beta, void* alpha);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_intarrayref_tensoroptions)(void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_tensor_tensor_tensoroptions)(void* indices, void* values, void* options);
LANTERN_API void* (LANTERN_PTR lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions)(void* indices, void* values, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions)(void* indices, void* values, void* size, void* options);
LANTERN_API void* (LANTERN_PTR lantern_to_dense_backward_tensor_tensor)(void* grad, void* input);
LANTERN_API void* (LANTERN_PTR lantern_to_mkldnn_backward_tensor_tensor)(void* grad, void* input);
LANTERN_API void* (LANTERN_PTR lantern_meshgrid_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_cartesian_prod_tensorlist)(void* tensors);
LANTERN_API void* (LANTERN_PTR lantern_combinations_tensor_intt_bool)(void* self, void* r, void* with_replacement);
LANTERN_API void* (LANTERN_PTR lantern_result_type_tensor_tensor)(void* tensor, void* other);
LANTERN_API void* (LANTERN_PTR lantern_result_type_tensor_scalar)(void* tensor, void* other);
LANTERN_API void* (LANTERN_PTR lantern_result_type_scalar_tensor)(void* scalar, void* tensor);
LANTERN_API void* (LANTERN_PTR lantern_result_type_scalar_scalar)(void* scalar1, void* scalar2);
LANTERN_API void* (LANTERN_PTR lantern_can_cast_scalartype_scalartype)(void* from, void* to);
LANTERN_API void* (LANTERN_PTR lantern_promote_types_scalartype_scalartype)(void* type1, void* type2);
LANTERN_API void* (LANTERN_PTR lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh);
LANTERN_API void* (LANTERN_PTR lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool)(void* grad, void* input_size, void* batch_sizes, void* batch_first);
LANTERN_API void* (LANTERN_PTR lantern_cross_out_tensor_tensor_tensor_intt)(void* out, void* self, void* other, void* dim);
LANTERN_API void* (LANTERN_PTR lantern_index_select_out_tensor_tensor_dimname_tensor)(void* out, void* self, void* dim, void* index);
LANTERN_API void* (LANTERN_PTR lantern_gather_out_tensor_tensor_dimname_tensor_bool)(void* out, void* self, void* dim, void* index, void* sparse_grad);
LANTERN_API void* (LANTERN_PTR lantern__gather_sparse_backward_tensor_intt_tensor_tensor)(void* self, void* dim, void* index, void* grad);
LANTERN_API void* (LANTERN_PTR lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar)(void* out, void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar)(void* out, void* self, void* tensor1, void* tensor2, void* value);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_out_tensor_tensor_bool)(void* out, void* self, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_cholesky_solve_out_tensor_tensor_tensor_bool)(void* out, void* self, void* input2, void* upper);
LANTERN_API void* (LANTERN_PTR lantern_lu_solve_out_tensor_tensor_tensor_tensor)(void* out, void* self, void* LU_data, void* LU_pivots);
LANTERN_API void* (LANTERN_PTR lantern_digamma_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_polygamma_out_tensor_intt_tensor)(void* out, void* n, void* self);
LANTERN_API void* (LANTERN_PTR lantern_atan2_out_tensor_tensor_tensor)(void* out, void* self, void* other);
LANTERN_API void* (LANTERN_PTR lantern_normal_double_double_intarrayref_generator_tensoroptions)(void* mean, void* std, void* size, void* generator, void* options);
LANTERN_API void* (LANTERN_PTR lantern_normal_out_tensor_double_double_intarrayref_generator)(void* out, void* mean, void* std, void* size, void* generator);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt)(void* out, void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_multilabel_margin_loss_tensor_tensor_intt)(void* self, void* target, void* reduction);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt)(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt)(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_nll_loss2d_tensor_tensor_tensor_intt_intt)(void* self, void* target, void* weight, void* reduction, void* ignore_index);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_out_tensor_tensor)(void* out, void* self);
LANTERN_API void* (LANTERN_PTR lantern_log_sigmoid_tensor)(void* self);
LANTERN_API void* (LANTERN_PTR lantern_adaptive_avg_pool2d_tensor_intarrayref)(void* self, void* output_size);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
LANTERN_API void* (LANTERN_PTR lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding);
/* Autogen Headers -- End */
  
#ifdef __cplusplus
}
#endif

#ifndef LANTERN_BUILD

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
  LOAD_SYMBOL(lantern_align_tensors_tensorlist)
  LOAD_SYMBOL(lantern__debug_has_internal_overlap_tensor)
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
  LOAD_SYMBOL(lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_adaptive_avg_pool1d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_affine_grid_generator_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_affine_grid_generator_backward_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_all_out_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_all_out_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_any_out_tensor_tensor_intt_bool)
  LOAD_SYMBOL(lantern_any_out_tensor_tensor_dimname_bool)
  LOAD_SYMBOL(lantern_arange_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_scalar_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_arange_out_tensor_scalar)
  LOAD_SYMBOL(lantern__dim_arange_tensor_intt)
  LOAD_SYMBOL(lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_bartlett_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_bartlett_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)
  LOAD_SYMBOL(lantern_bernoulli_out_tensor_tensor_generator)
  LOAD_SYMBOL(lantern_bilinear_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_blackman_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_blackman_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_broadcast_tensors_tensorlist)
  LOAD_SYMBOL(lantern_cat_tensorlist_intt)
  LOAD_SYMBOL(lantern_cat_out_tensor_tensorlist_intt)
  LOAD_SYMBOL(lantern_cat_tensorlist_dimname)
  LOAD_SYMBOL(lantern_cat_out_tensor_tensorlist_dimname)
  LOAD_SYMBOL(lantern_chain_matmul_tensorlist)
  LOAD_SYMBOL(lantern_cudnn_is_acceptable_tensor)
  LOAD_SYMBOL(lantern_constant_pad_nd_tensor_intarrayref_scalar)
  LOAD_SYMBOL(lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)
  LOAD_SYMBOL(lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt)
  LOAD_SYMBOL(lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool)
  LOAD_SYMBOL(lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref)
  LOAD_SYMBOL(lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_conv_tbc_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref)
  LOAD_SYMBOL(lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_cumsum_out_tensor_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumsum_out_tensor_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_cumprod_out_tensor_tensor_intt_scalartype)
  LOAD_SYMBOL(lantern_cumprod_out_tensor_tensor_dimname_scalartype)
  LOAD_SYMBOL(lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool)
  LOAD_SYMBOL(lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_dot_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_einsum_stdstring_tensorlist)
  LOAD_SYMBOL(lantern_embedding_tensor_tensor_intt_bool_bool)
  LOAD_SYMBOL(lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool)
  LOAD_SYMBOL(lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor)
  LOAD_SYMBOL(lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor)
  LOAD_SYMBOL(lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_empty_out_tensor_intarrayref_memoryformat)
  LOAD_SYMBOL(lantern_empty_like_tensor)
  LOAD_SYMBOL(lantern_empty_like_tensor_tensoroptions_memoryformat)
  LOAD_SYMBOL(lantern_eye_intt_tensoroptions)
  LOAD_SYMBOL(lantern_eye_intt_intt_tensoroptions)
  LOAD_SYMBOL(lantern_full_intarrayref_scalar_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_full_intarrayref_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_full_out_tensor_intarrayref_scalar)
  LOAD_SYMBOL(lantern_full_like_tensor_scalar)
  LOAD_SYMBOL(lantern_full_like_tensor_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_grid_sampler_tensor_tensor_intt_intt_bool)
  LOAD_SYMBOL(lantern_hann_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_hann_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_double_tensoroptions)
  LOAD_SYMBOL(lantern_hamming_window_intt_bool_double_double_tensoroptions)
  LOAD_SYMBOL(lantern_hinge_embedding_loss_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_group_norm_tensor_intt_tensor_tensor_double_bool)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_size_intt)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_max_size_intt)
  LOAD_SYMBOL(lantern__cufft_set_plan_cache_max_size_intt_intt)
  LOAD_SYMBOL(lantern__cufft_clear_plan_cache_intt)
  LOAD_SYMBOL(lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool)
  LOAD_SYMBOL(lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool)
  LOAD_SYMBOL(lantern_inverse_out_tensor_tensor)
  LOAD_SYMBOL(lantern_isnan_tensor)
  LOAD_SYMBOL(lantern_kl_div_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool)
  LOAD_SYMBOL(lantern_linear_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_gemm_matrix_fp16_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix_tensor)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt)
  LOAD_SYMBOL(lantern_linspace_scalar_scalar_intt_tensoroptions)
  LOAD_SYMBOL(lantern_logspace_scalar_scalar_intt_double_tensoroptions)
  LOAD_SYMBOL(lantern_logsumexp_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_logsumexp_out_tensor_tensor_dimnamelist_bool)
  LOAD_SYMBOL(lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt)
  LOAD_SYMBOL(lantern_matmul_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_matrix_rank_tensor_double_bool)
  LOAD_SYMBOL(lantern_matrix_rank_tensor_bool)
  LOAD_SYMBOL(lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool)
  LOAD_SYMBOL(lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool)
  LOAD_SYMBOL(lantern__sparse_mm_tensor_tensor)
  LOAD_SYMBOL(lantern__nnpack_available)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref)
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
  LOAD_SYMBOL(lantern_pixel_shuffle_tensor_intt)
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
  LOAD_SYMBOL(lantern_range_scalar_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_range_scalar_scalar_tensoroptions)
  LOAD_SYMBOL(lantern_rrelu_tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_rrelu__tensor_scalar_scalar_bool_generator)
  LOAD_SYMBOL(lantern_selu_tensor)
  LOAD_SYMBOL(lantern_selu__tensor)
  LOAD_SYMBOL(lantern_celu_tensor_scalar)
  LOAD_SYMBOL(lantern_celu__tensor_scalar)
  LOAD_SYMBOL(lantern_stack_tensorlist_intt)
  LOAD_SYMBOL(lantern_stack_out_tensor_tensorlist_intt)
  LOAD_SYMBOL(lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_std_out_tensor_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_std_out_tensor_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_prod_out_tensor_tensor_intt_bool_scalartype)
  LOAD_SYMBOL(lantern_prod_out_tensor_tensor_dimname_bool_scalartype)
  LOAD_SYMBOL(lantern_tensordot_tensor_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_threshold_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold__tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold_out_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_threshold_backward_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_one_hot_tensor_intt)
  LOAD_SYMBOL(lantern_trapz_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_trapz_tensor_double_intt)
  LOAD_SYMBOL(lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt)
  LOAD_SYMBOL(lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt)
  LOAD_SYMBOL(lantern__has_compatible_shallow_copy_type_tensor_tensor)
  LOAD_SYMBOL(lantern__unsafe_view_tensor_intarrayref)
  LOAD_SYMBOL(lantern_var_out_tensor_tensor_intarrayref_bool_bool)
  LOAD_SYMBOL(lantern_var_out_tensor_tensor_dimnamelist_bool_bool)
  LOAD_SYMBOL(lantern_where_tensor)
  LOAD_SYMBOL(lantern_norm_except_dim_tensor_intt_intt)
  LOAD_SYMBOL(lantern__weight_norm_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_zeros_intarrayref_dimnamelist_tensoroptions)
  LOAD_SYMBOL(lantern_zeros_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_zeros_out_tensor_intarrayref)
  LOAD_SYMBOL(lantern_zeros_like_tensor)
  LOAD_SYMBOL(lantern_zeros_like_tensor_tensoroptions)
  LOAD_SYMBOL(lantern__sparse_sum_tensor)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_scalartype)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_intarrayref)
  LOAD_SYMBOL(lantern__sparse_sum_tensor_intarrayref_scalartype)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_intarrayref_bool)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype)
  LOAD_SYMBOL(lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool)
  LOAD_SYMBOL(lantern_frobenius_norm_tensor)
  LOAD_SYMBOL(lantern_frobenius_norm_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_tensor_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_out_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool)
  LOAD_SYMBOL(lantern_rsub_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_rsub_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_tensor_tensor_tensoroptions)
  LOAD_SYMBOL(lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions)
  LOAD_SYMBOL(lantern_to_dense_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_to_mkldnn_backward_tensor_tensor)
  LOAD_SYMBOL(lantern_meshgrid_tensorlist)
  LOAD_SYMBOL(lantern_cartesian_prod_tensorlist)
  LOAD_SYMBOL(lantern_combinations_tensor_intt_bool)
  LOAD_SYMBOL(lantern_result_type_tensor_tensor)
  LOAD_SYMBOL(lantern_result_type_tensor_scalar)
  LOAD_SYMBOL(lantern_result_type_scalar_tensor)
  LOAD_SYMBOL(lantern_result_type_scalar_scalar)
  LOAD_SYMBOL(lantern_can_cast_scalartype_scalartype)
  LOAD_SYMBOL(lantern_promote_types_scalartype_scalartype)
  LOAD_SYMBOL(lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar)
  LOAD_SYMBOL(lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool)
  LOAD_SYMBOL(lantern_cross_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_index_select_out_tensor_tensor_dimname_tensor)
  LOAD_SYMBOL(lantern_gather_out_tensor_tensor_dimname_tensor_bool)
  LOAD_SYMBOL(lantern__gather_sparse_backward_tensor_intt_tensor_tensor)
  LOAD_SYMBOL(lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar)
  LOAD_SYMBOL(lantern_cholesky_out_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_cholesky_solve_out_tensor_tensor_tensor_bool)
  LOAD_SYMBOL(lantern_lu_solve_out_tensor_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_digamma_out_tensor_tensor)
  LOAD_SYMBOL(lantern_polygamma_out_tensor_intt_tensor)
  LOAD_SYMBOL(lantern_atan2_out_tensor_tensor_tensor)
  LOAD_SYMBOL(lantern_normal_double_double_intarrayref_generator_tensoroptions)
  LOAD_SYMBOL(lantern_normal_out_tensor_double_double_intarrayref_generator)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_tensor_tensor_intt)
  LOAD_SYMBOL(lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_nll_loss2d_tensor_tensor_tensor_intt_intt)
  LOAD_SYMBOL(lantern_log_sigmoid_out_tensor_tensor)
  LOAD_SYMBOL(lantern_log_sigmoid_tensor)
  LOAD_SYMBOL(lantern_adaptive_avg_pool2d_tensor_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  LOAD_SYMBOL(lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref)
  /* Autogen Symbols -- End */
  
  return true;
}

#endif
#endif
