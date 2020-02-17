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
/*
LANTERN_API void (LANTERN_PTR lantern__cast_byte)();
LANTERN_API void (LANTERN_PTR lantern__cast_char)();
LANTERN_API void (LANTERN_PTR lantern__cast_double)();
LANTERN_API void (LANTERN_PTR lantern__cast_float)();
LANTERN_API void (LANTERN_PTR lantern__cast_int)();
LANTERN_API void (LANTERN_PTR lantern__cast_long)();
LANTERN_API void (LANTERN_PTR lantern__cast_short)();
LANTERN_API void (LANTERN_PTR lantern__cast_half)();
LANTERN_API void (LANTERN_PTR lantern_backward)();
LANTERN_API void (LANTERN_PTR lantern_set_data)();
LANTERN_API void (LANTERN_PTR lantern_data)();
LANTERN_API void (LANTERN_PTR lantern_is_leaf)();
LANTERN_API void (LANTERN_PTR lantern_output_nr)();
LANTERN_API void (LANTERN_PTR lantern__version)();
LANTERN_API void (LANTERN_PTR lantern_rename_)();
LANTERN_API void (LANTERN_PTR lantern_rename)();
LANTERN_API void (LANTERN_PTR lantern_align_to)();
LANTERN_API void (LANTERN_PTR lantern_align_as)();
LANTERN_API void (LANTERN_PTR lantern_align_tensors)();
LANTERN_API void (LANTERN_PTR lantern_refine_names)();
LANTERN_API void (LANTERN_PTR lantern_unflatten)();
LANTERN_API void (LANTERN_PTR lantern_unflatten)();
LANTERN_API void (LANTERN_PTR lantern__cudnn_ctc_loss)();
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn_flatten_weight)();
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn)();
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn_backward)();
LANTERN_API void (LANTERN_PTR lantern__cudnn_init_dropout_state)();
LANTERN_API void (LANTERN_PTR lantern__debug_has_internal_overlap)();
LANTERN_API void (LANTERN_PTR lantern__fused_dropout)();
LANTERN_API void (LANTERN_PTR lantern__masked_scale)();
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_draw)();
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_ff_)();
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_scramble_)();
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_initialize_state_)();
LANTERN_API void (LANTERN_PTR lantern__reshape_from_tensor)();
LANTERN_API void (LANTERN_PTR lantern__shape_as_tensor)();
LANTERN_API void (LANTERN_PTR lantern_dropout)();
LANTERN_API void (LANTERN_PTR lantern_dropout_)();
LANTERN_API void (LANTERN_PTR lantern_feature_dropout)();
LANTERN_API void (LANTERN_PTR lantern_feature_dropout_)();
LANTERN_API void (LANTERN_PTR lantern_alpha_dropout)();
LANTERN_API void (LANTERN_PTR lantern_alpha_dropout_)();
LANTERN_API void (LANTERN_PTR lantern_feature_alpha_dropout)();
LANTERN_API void (LANTERN_PTR lantern_feature_alpha_dropout_)();
LANTERN_API void (LANTERN_PTR lantern_abs)();
LANTERN_API void (LANTERN_PTR lantern_abs_)();
LANTERN_API void (LANTERN_PTR lantern_abs_out)();
LANTERN_API void (LANTERN_PTR lantern_acos)();
LANTERN_API void (LANTERN_PTR lantern_acos_)();
LANTERN_API void (LANTERN_PTR lantern_acos_out)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool1d)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool1d)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool1d)();
LANTERN_API void (LANTERN_PTR lantern_add)();
LANTERN_API void (LANTERN_PTR lantern_add_)();
LANTERN_API void (LANTERN_PTR lantern_add_out)();
LANTERN_API void (LANTERN_PTR lantern_add)();
LANTERN_API void (LANTERN_PTR lantern_add_)();
LANTERN_API void (LANTERN_PTR lantern_addmv)();
LANTERN_API void (LANTERN_PTR lantern_addmv_)();
LANTERN_API void (LANTERN_PTR lantern_addmv_out)();
LANTERN_API void (LANTERN_PTR lantern_addr)();
LANTERN_API void (LANTERN_PTR lantern_addr_)();
LANTERN_API void (LANTERN_PTR lantern_addr_out)();
LANTERN_API void (LANTERN_PTR lantern_affine_grid_generator)();
LANTERN_API void (LANTERN_PTR lantern_affine_grid_generator_backward)();
LANTERN_API void (LANTERN_PTR lantern_all)();
LANTERN_API void (LANTERN_PTR lantern_all_out)();
LANTERN_API void (LANTERN_PTR lantern_all)();
LANTERN_API void (LANTERN_PTR lantern_all_out)();
LANTERN_API void (LANTERN_PTR lantern_allclose)();
LANTERN_API void (LANTERN_PTR lantern_any)();
LANTERN_API void (LANTERN_PTR lantern_any_out)();
LANTERN_API void (LANTERN_PTR lantern_any)();
LANTERN_API void (LANTERN_PTR lantern_any_out)();
LANTERN_API void (LANTERN_PTR lantern_arange)();
LANTERN_API void (LANTERN_PTR lantern_arange)();
LANTERN_API void (LANTERN_PTR lantern_arange)();
LANTERN_API void (LANTERN_PTR lantern_arange_out)();
LANTERN_API void (LANTERN_PTR lantern_arange_out)();
LANTERN_API void (LANTERN_PTR lantern__dim_arange)();
LANTERN_API void (LANTERN_PTR lantern_argmax)();
LANTERN_API void (LANTERN_PTR lantern_argmin)();
LANTERN_API void (LANTERN_PTR lantern_as_strided)();
LANTERN_API void (LANTERN_PTR lantern_as_strided_)();
LANTERN_API void (LANTERN_PTR lantern_asin)();
LANTERN_API void (LANTERN_PTR lantern_asin_)();
LANTERN_API void (LANTERN_PTR lantern_asin_out)();
LANTERN_API void (LANTERN_PTR lantern_atan)();
LANTERN_API void (LANTERN_PTR lantern_atan_)();
LANTERN_API void (LANTERN_PTR lantern_atan_out)();
LANTERN_API void (LANTERN_PTR lantern_baddbmm)();
LANTERN_API void (LANTERN_PTR lantern_baddbmm_)();
LANTERN_API void (LANTERN_PTR lantern__baddbmm_mkl_)();
LANTERN_API void (LANTERN_PTR lantern_baddbmm_out)();
LANTERN_API void (LANTERN_PTR lantern_bartlett_window)();
LANTERN_API void (LANTERN_PTR lantern_bartlett_window)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm)();
LANTERN_API void (LANTERN_PTR lantern__batch_norm_impl_index)();
LANTERN_API void (LANTERN_PTR lantern__batch_norm_impl_index_backward)();
LANTERN_API void (LANTERN_PTR lantern_bernoulli)();
LANTERN_API void (LANTERN_PTR lantern_bernoulli_out)();
LANTERN_API void (LANTERN_PTR lantern_bernoulli_)();
LANTERN_API void (LANTERN_PTR lantern_bernoulli_)();
LANTERN_API void (LANTERN_PTR lantern_bernoulli)();
LANTERN_API void (LANTERN_PTR lantern_bilinear)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_with_logits)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_with_logits_backward)();
LANTERN_API void (LANTERN_PTR lantern_bincount)();
LANTERN_API void (LANTERN_PTR lantern_bitwise_not)();
LANTERN_API void (LANTERN_PTR lantern_bitwise_not_)();
LANTERN_API void (LANTERN_PTR lantern_bitwise_not_out)();
LANTERN_API void (LANTERN_PTR lantern_logical_not)();
LANTERN_API void (LANTERN_PTR lantern_logical_not_)();
LANTERN_API void (LANTERN_PTR lantern_logical_not_out)();
LANTERN_API void (LANTERN_PTR lantern_logical_xor)();
LANTERN_API void (LANTERN_PTR lantern_logical_xor_)();
LANTERN_API void (LANTERN_PTR lantern_logical_xor_out)();
LANTERN_API void (LANTERN_PTR lantern_blackman_window)();
LANTERN_API void (LANTERN_PTR lantern_blackman_window)();
LANTERN_API void (LANTERN_PTR lantern_bmm)();
LANTERN_API void (LANTERN_PTR lantern_bmm_out)();
LANTERN_API void (LANTERN_PTR lantern_broadcast_tensors)();
LANTERN_API void (LANTERN_PTR lantern_cat)();
LANTERN_API void (LANTERN_PTR lantern_cat_out)();
LANTERN_API void (LANTERN_PTR lantern_cat)();
LANTERN_API void (LANTERN_PTR lantern_cat_out)();
LANTERN_API void (LANTERN_PTR lantern_ceil)();
LANTERN_API void (LANTERN_PTR lantern_ceil_)();
LANTERN_API void (LANTERN_PTR lantern_ceil_out)();
LANTERN_API void (LANTERN_PTR lantern_chain_matmul)();
LANTERN_API void (LANTERN_PTR lantern_chunk)();
LANTERN_API void (LANTERN_PTR lantern_clamp)();
LANTERN_API void (LANTERN_PTR lantern_clamp_)();
LANTERN_API void (LANTERN_PTR lantern_clamp_out)();
LANTERN_API void (LANTERN_PTR lantern_clamp_max)();
LANTERN_API void (LANTERN_PTR lantern_clamp_max_)();
LANTERN_API void (LANTERN_PTR lantern_clamp_max_out)();
LANTERN_API void (LANTERN_PTR lantern_clamp_min)();
LANTERN_API void (LANTERN_PTR lantern_clamp_min_)();
LANTERN_API void (LANTERN_PTR lantern_clamp_min_out)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_is_acceptable)();
LANTERN_API void (LANTERN_PTR lantern_constant_pad_nd)();
LANTERN_API void (LANTERN_PTR lantern_contiguous)();
LANTERN_API void (LANTERN_PTR lantern_convolution)();
LANTERN_API void (LANTERN_PTR lantern_convolution_overrideable)();
LANTERN_API void (LANTERN_PTR lantern_convolution_backward_overrideable)();
LANTERN_API void (LANTERN_PTR lantern__convolution)();
LANTERN_API void (LANTERN_PTR lantern__convolution_nogroup)();
LANTERN_API void (LANTERN_PTR lantern__convolution_double_backward)();
LANTERN_API void (LANTERN_PTR lantern_conv1d)();
LANTERN_API void (LANTERN_PTR lantern_conv2d)();
LANTERN_API void (LANTERN_PTR lantern_conv3d)();
LANTERN_API void (LANTERN_PTR lantern_conv_tbc)();
LANTERN_API void (LANTERN_PTR lantern_conv_tbc_backward)();
LANTERN_API void (LANTERN_PTR lantern_conv_transpose1d)();
LANTERN_API void (LANTERN_PTR lantern_conv_transpose2d)();
LANTERN_API void (LANTERN_PTR lantern_conv_transpose3d)();
LANTERN_API void (LANTERN_PTR lantern_copy_)();
LANTERN_API void (LANTERN_PTR lantern__copy_from)();
LANTERN_API void (LANTERN_PTR lantern_cos)();
LANTERN_API void (LANTERN_PTR lantern_cos_)();
LANTERN_API void (LANTERN_PTR lantern_cos_out)();
LANTERN_API void (LANTERN_PTR lantern_cosh)();
LANTERN_API void (LANTERN_PTR lantern_cosh_)();
LANTERN_API void (LANTERN_PTR lantern_cosh_out)();
LANTERN_API void (LANTERN_PTR lantern_cosine_embedding_loss)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_affine_grid_generator)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_affine_grid_generator_backward)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_batch_norm)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_bias)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_bias)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_grid_sampler)();
LANTERN_API void (LANTERN_PTR lantern_cudnn_grid_sampler_backward)();
LANTERN_API void (LANTERN_PTR lantern_cumsum)();
LANTERN_API void (LANTERN_PTR lantern_cumsum_out)();
LANTERN_API void (LANTERN_PTR lantern_cumsum)();
LANTERN_API void (LANTERN_PTR lantern_cumsum_out)();
LANTERN_API void (LANTERN_PTR lantern_cumprod)();
LANTERN_API void (LANTERN_PTR lantern_cumprod_out)();
LANTERN_API void (LANTERN_PTR lantern_cumprod)();
LANTERN_API void (LANTERN_PTR lantern_cumprod_out)();
LANTERN_API void (LANTERN_PTR lantern_ctc_loss)();
LANTERN_API void (LANTERN_PTR lantern_ctc_loss)();
LANTERN_API void (LANTERN_PTR lantern__ctc_loss)();
LANTERN_API void (LANTERN_PTR lantern__ctc_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_det)();
LANTERN_API void (LANTERN_PTR lantern_diag_embed)();
LANTERN_API void (LANTERN_PTR lantern_diagflat)();
LANTERN_API void (LANTERN_PTR lantern_diagonal)();
LANTERN_API void (LANTERN_PTR lantern_fill_diagonal_)();
LANTERN_API void (LANTERN_PTR lantern_div)();
LANTERN_API void (LANTERN_PTR lantern_div_)();
LANTERN_API void (LANTERN_PTR lantern_div_out)();
LANTERN_API void (LANTERN_PTR lantern_div)();
LANTERN_API void (LANTERN_PTR lantern_div_)();
LANTERN_API void (LANTERN_PTR lantern_dot)();
LANTERN_API void (LANTERN_PTR lantern_dot_out)();
LANTERN_API void (LANTERN_PTR lantern_einsum)();
LANTERN_API void (LANTERN_PTR lantern_embedding)();
LANTERN_API void (LANTERN_PTR lantern_embedding_backward)();
LANTERN_API void (LANTERN_PTR lantern_embedding_dense_backward)();
LANTERN_API void (LANTERN_PTR lantern_embedding_renorm_)();
LANTERN_API void (LANTERN_PTR lantern_embedding_sparse_backward)();
LANTERN_API void (LANTERN_PTR lantern_embedding_bag)();
LANTERN_API void (LANTERN_PTR lantern__embedding_bag)();
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_backward)();
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_sparse_backward)();
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_dense_backward)();
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_per_sample_weights_backward)();
LANTERN_API void (LANTERN_PTR lantern_empty)();
LANTERN_API void (LANTERN_PTR lantern_empty)();
LANTERN_API void (LANTERN_PTR lantern_new_empty)();
LANTERN_API void (LANTERN_PTR lantern_new_full)();
LANTERN_API void (LANTERN_PTR lantern__empty_affine_quantized)();
LANTERN_API void (LANTERN_PTR lantern__empty_per_channel_affine_quantized)();
LANTERN_API void (LANTERN_PTR lantern_resize_)();
LANTERN_API void (LANTERN_PTR lantern_empty_out)();
LANTERN_API void (LANTERN_PTR lantern_empty_like)();
LANTERN_API void (LANTERN_PTR lantern_empty_like)();
LANTERN_API void (LANTERN_PTR lantern_empty_strided)();
LANTERN_API void (LANTERN_PTR lantern_erf)();
LANTERN_API void (LANTERN_PTR lantern_erf_)();
LANTERN_API void (LANTERN_PTR lantern_erf_out)();
LANTERN_API void (LANTERN_PTR lantern_erfc)();
LANTERN_API void (LANTERN_PTR lantern_erfc_)();
LANTERN_API void (LANTERN_PTR lantern_erfc_out)();
LANTERN_API void (LANTERN_PTR lantern_exp)();
LANTERN_API void (LANTERN_PTR lantern_exp_)();
LANTERN_API void (LANTERN_PTR lantern_exp_out)();
LANTERN_API void (LANTERN_PTR lantern_expm1)();
LANTERN_API void (LANTERN_PTR lantern_expm1_)();
LANTERN_API void (LANTERN_PTR lantern_expm1_out)();
LANTERN_API void (LANTERN_PTR lantern_expand)();
LANTERN_API void (LANTERN_PTR lantern_expand_as)();
LANTERN_API void (LANTERN_PTR lantern_eye)();
LANTERN_API void (LANTERN_PTR lantern_eye)();
LANTERN_API void (LANTERN_PTR lantern_eye_out)();
LANTERN_API void (LANTERN_PTR lantern_eye_out)();
LANTERN_API void (LANTERN_PTR lantern_flatten)();
LANTERN_API void (LANTERN_PTR lantern_flatten)();
LANTERN_API void (LANTERN_PTR lantern_flatten)();
LANTERN_API void (LANTERN_PTR lantern_flatten)();
LANTERN_API void (LANTERN_PTR lantern_fill_)();
LANTERN_API void (LANTERN_PTR lantern_fill_)();
LANTERN_API void (LANTERN_PTR lantern_floor)();
LANTERN_API void (LANTERN_PTR lantern_floor_)();
LANTERN_API void (LANTERN_PTR lantern_floor_out)();
LANTERN_API void (LANTERN_PTR lantern_frac)();
LANTERN_API void (LANTERN_PTR lantern_frac_)();
LANTERN_API void (LANTERN_PTR lantern_frac_out)();
LANTERN_API void (LANTERN_PTR lantern_full)();
LANTERN_API void (LANTERN_PTR lantern_full)();
LANTERN_API void (LANTERN_PTR lantern_full_out)();
LANTERN_API void (LANTERN_PTR lantern_full_like)();
LANTERN_API void (LANTERN_PTR lantern_full_like)();
LANTERN_API void (LANTERN_PTR lantern_from_file)();
LANTERN_API void (LANTERN_PTR lantern_grid_sampler)();
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_2d)();
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_3d)();
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_hann_window)();
LANTERN_API void (LANTERN_PTR lantern_hann_window)();
LANTERN_API void (LANTERN_PTR lantern_hamming_window)();
LANTERN_API void (LANTERN_PTR lantern_hamming_window)();
LANTERN_API void (LANTERN_PTR lantern_hamming_window)();
LANTERN_API void (LANTERN_PTR lantern_hamming_window)();
LANTERN_API void (LANTERN_PTR lantern_hinge_embedding_loss)();
LANTERN_API void (LANTERN_PTR lantern_ger)();
LANTERN_API void (LANTERN_PTR lantern_ger_out)();
LANTERN_API void (LANTERN_PTR lantern_group_norm)();
LANTERN_API void (LANTERN_PTR lantern_fft)();
LANTERN_API void (LANTERN_PTR lantern_ifft)();
LANTERN_API void (LANTERN_PTR lantern_rfft)();
LANTERN_API void (LANTERN_PTR lantern_irfft)();
LANTERN_API void (LANTERN_PTR lantern__fft_with_size)();
LANTERN_API void (LANTERN_PTR lantern__cufft_get_plan_cache_size)();
LANTERN_API void (LANTERN_PTR lantern__cufft_get_plan_cache_max_size)();
LANTERN_API void (LANTERN_PTR lantern__cufft_set_plan_cache_max_size)();
LANTERN_API void (LANTERN_PTR lantern__cufft_clear_plan_cache)();
LANTERN_API void (LANTERN_PTR lantern_index)();
LANTERN_API void (LANTERN_PTR lantern_index_copy_)();
LANTERN_API void (LANTERN_PTR lantern_index_copy)();
LANTERN_API void (LANTERN_PTR lantern_index_copy_)();
LANTERN_API void (LANTERN_PTR lantern_index_copy)();
LANTERN_API void (LANTERN_PTR lantern_index_put_)();
LANTERN_API void (LANTERN_PTR lantern_index_put)();
LANTERN_API void (LANTERN_PTR lantern__index_put_impl_)();
LANTERN_API void (LANTERN_PTR lantern_instance_norm)();
LANTERN_API void (LANTERN_PTR lantern_inverse)();
LANTERN_API void (LANTERN_PTR lantern_inverse_out)();
LANTERN_API void (LANTERN_PTR lantern__inverse_helper)();
LANTERN_API void (LANTERN_PTR lantern_isclose)();
LANTERN_API void (LANTERN_PTR lantern_isnan)();
LANTERN_API void (LANTERN_PTR lantern_is_distributed)();
LANTERN_API void (LANTERN_PTR lantern_is_floating_point)();
LANTERN_API void (LANTERN_PTR lantern_is_complex)();
LANTERN_API void (LANTERN_PTR lantern_is_nonzero)();
LANTERN_API void (LANTERN_PTR lantern_is_same_size)();
LANTERN_API void (LANTERN_PTR lantern_is_signed)();
LANTERN_API void (LANTERN_PTR lantern_kl_div)();
LANTERN_API void (LANTERN_PTR lantern_kl_div_backward)();
LANTERN_API void (LANTERN_PTR lantern_kthvalue)();
LANTERN_API void (LANTERN_PTR lantern_kthvalue_out)();
LANTERN_API void (LANTERN_PTR lantern_kthvalue)();
LANTERN_API void (LANTERN_PTR lantern_kthvalue_out)();
LANTERN_API void (LANTERN_PTR lantern_layer_norm)();
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm)();
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm_backward)();
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm_double_backward)();
LANTERN_API void (LANTERN_PTR lantern_linear)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_linear)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_int8_weight_fp32_activation)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_int8_weight)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_quantize_weight)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_gemm_matrix_fp16)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_fp32_activation)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_fp16_weight)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix)();
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix)();
LANTERN_API void (LANTERN_PTR lantern_linspace)();
LANTERN_API void (LANTERN_PTR lantern_linspace_out)();
LANTERN_API void (LANTERN_PTR lantern_log)();
LANTERN_API void (LANTERN_PTR lantern_log_)();
LANTERN_API void (LANTERN_PTR lantern_log_out)();
LANTERN_API void (LANTERN_PTR lantern_log10)();
LANTERN_API void (LANTERN_PTR lantern_log10_)();
LANTERN_API void (LANTERN_PTR lantern_log10_out)();
LANTERN_API void (LANTERN_PTR lantern_log1p)();
LANTERN_API void (LANTERN_PTR lantern_log1p_)();
LANTERN_API void (LANTERN_PTR lantern_log1p_out)();
LANTERN_API void (LANTERN_PTR lantern_log2)();
LANTERN_API void (LANTERN_PTR lantern_log2_)();
LANTERN_API void (LANTERN_PTR lantern_log2_out)();
LANTERN_API void (LANTERN_PTR lantern_logdet)();
LANTERN_API void (LANTERN_PTR lantern_logspace)();
LANTERN_API void (LANTERN_PTR lantern_logspace_out)();
LANTERN_API void (LANTERN_PTR lantern_log_softmax)();
LANTERN_API void (LANTERN_PTR lantern_log_softmax)();
LANTERN_API void (LANTERN_PTR lantern__log_softmax)();
LANTERN_API void (LANTERN_PTR lantern__log_softmax_backward_data)();
LANTERN_API void (LANTERN_PTR lantern_logsumexp)();
LANTERN_API void (LANTERN_PTR lantern_logsumexp_out)();
LANTERN_API void (LANTERN_PTR lantern_logsumexp)();
LANTERN_API void (LANTERN_PTR lantern_logsumexp_out)();
LANTERN_API void (LANTERN_PTR lantern_margin_ranking_loss)();
LANTERN_API void (LANTERN_PTR lantern_matmul)();
LANTERN_API void (LANTERN_PTR lantern_matmul_out)();
LANTERN_API void (LANTERN_PTR lantern_matrix_rank)();
LANTERN_API void (LANTERN_PTR lantern_matrix_rank)();
LANTERN_API void (LANTERN_PTR lantern_matrix_power)();
LANTERN_API void (LANTERN_PTR lantern_max)();
LANTERN_API void (LANTERN_PTR lantern_max_out)();
LANTERN_API void (LANTERN_PTR lantern_max_values)();
LANTERN_API void (LANTERN_PTR lantern_max)();
LANTERN_API void (LANTERN_PTR lantern_max_out)();
LANTERN_API void (LANTERN_PTR lantern_max_values)();
LANTERN_API void (LANTERN_PTR lantern_max_pool1d_with_indices)();
LANTERN_API void (LANTERN_PTR lantern_max_pool1d)();
LANTERN_API void (LANTERN_PTR lantern_max_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_max_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_quantized_max_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_max_pool3d)();
LANTERN_API void (LANTERN_PTR lantern_mean)();
LANTERN_API void (LANTERN_PTR lantern_mean)();
LANTERN_API void (LANTERN_PTR lantern_mean_out)();
LANTERN_API void (LANTERN_PTR lantern_mean)();
LANTERN_API void (LANTERN_PTR lantern_mean_out)();
LANTERN_API void (LANTERN_PTR lantern_median)();
LANTERN_API void (LANTERN_PTR lantern_median_out)();
LANTERN_API void (LANTERN_PTR lantern_median)();
LANTERN_API void (LANTERN_PTR lantern_median_out)();
LANTERN_API void (LANTERN_PTR lantern_min)();
LANTERN_API void (LANTERN_PTR lantern_min_out)();
LANTERN_API void (LANTERN_PTR lantern_min_values)();
LANTERN_API void (LANTERN_PTR lantern_min)();
LANTERN_API void (LANTERN_PTR lantern_min_out)();
LANTERN_API void (LANTERN_PTR lantern_min_values)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward_weights)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward)();
LANTERN_API void (LANTERN_PTR lantern_miopen_batch_norm)();
LANTERN_API void (LANTERN_PTR lantern_miopen_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_bias)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution)();
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward)();
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_miopen_rnn)();
LANTERN_API void (LANTERN_PTR lantern_miopen_rnn_backward)();
LANTERN_API void (LANTERN_PTR lantern_mm)();
LANTERN_API void (LANTERN_PTR lantern_mm_out)();
LANTERN_API void (LANTERN_PTR lantern__sparse_mm)();
LANTERN_API void (LANTERN_PTR lantern_mode)();
LANTERN_API void (LANTERN_PTR lantern_mode_out)();
LANTERN_API void (LANTERN_PTR lantern_mode)();
LANTERN_API void (LANTERN_PTR lantern_mode_out)();
LANTERN_API void (LANTERN_PTR lantern_mul)();
LANTERN_API void (LANTERN_PTR lantern_mul_)();
LANTERN_API void (LANTERN_PTR lantern_mul_out)();
LANTERN_API void (LANTERN_PTR lantern_mul)();
LANTERN_API void (LANTERN_PTR lantern_mul_)();
LANTERN_API void (LANTERN_PTR lantern_mv)();
LANTERN_API void (LANTERN_PTR lantern_mv_out)();
LANTERN_API void (LANTERN_PTR lantern_mvlgamma)();
LANTERN_API void (LANTERN_PTR lantern_mvlgamma_)();
LANTERN_API void (LANTERN_PTR lantern_narrow_copy)();
LANTERN_API void (LANTERN_PTR lantern_narrow)();
LANTERN_API void (LANTERN_PTR lantern_native_batch_norm)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_stats)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_elemt)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_gather_stats)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_gather_stats_with_counts)();
LANTERN_API void (LANTERN_PTR lantern_native_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_backward_reduce)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_backward_elemt)();
LANTERN_API void (LANTERN_PTR lantern_batch_norm_update_stats)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_available)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR lantern_ones)();
LANTERN_API void (LANTERN_PTR lantern_ones)();
LANTERN_API void (LANTERN_PTR lantern_ones_out)();
LANTERN_API void (LANTERN_PTR lantern_ones_like)();
LANTERN_API void (LANTERN_PTR lantern_ones_like)();
LANTERN_API void (LANTERN_PTR lantern_pairwise_distance)();
LANTERN_API void (LANTERN_PTR lantern_cdist)();
LANTERN_API void (LANTERN_PTR lantern__cdist_backward)();
LANTERN_API void (LANTERN_PTR lantern_pdist)();
LANTERN_API void (LANTERN_PTR lantern__pdist_forward)();
LANTERN_API void (LANTERN_PTR lantern__pdist_backward)();
LANTERN_API void (LANTERN_PTR lantern_cosine_similarity)();
LANTERN_API void (LANTERN_PTR lantern_permute)();
LANTERN_API void (LANTERN_PTR lantern_numpy_t)();
LANTERN_API void (LANTERN_PTR lantern_pixel_shuffle)();
LANTERN_API void (LANTERN_PTR lantern_is_pinned)();
LANTERN_API void (LANTERN_PTR lantern_pin_memory)();
LANTERN_API void (LANTERN_PTR lantern_pinverse)();
LANTERN_API void (LANTERN_PTR lantern_poisson_nll_loss)();
LANTERN_API void (LANTERN_PTR lantern_scalar_tensor)();
LANTERN_API void (LANTERN_PTR lantern_rand)();
LANTERN_API void (LANTERN_PTR lantern_rand)();
LANTERN_API void (LANTERN_PTR lantern_rand)();
LANTERN_API void (LANTERN_PTR lantern_rand)();
LANTERN_API void (LANTERN_PTR lantern_rand_out)();
LANTERN_API void (LANTERN_PTR lantern_rand_out)();
LANTERN_API void (LANTERN_PTR lantern_rand_like)();
LANTERN_API void (LANTERN_PTR lantern_rand_like)();
LANTERN_API void (LANTERN_PTR lantern_randint)();
LANTERN_API void (LANTERN_PTR lantern_randint)();
LANTERN_API void (LANTERN_PTR lantern_randint)();
LANTERN_API void (LANTERN_PTR lantern_randint)();
LANTERN_API void (LANTERN_PTR lantern_randint_out)();
LANTERN_API void (LANTERN_PTR lantern_randint_out)();
LANTERN_API void (LANTERN_PTR lantern_randint_out)();
LANTERN_API void (LANTERN_PTR lantern_randint_out)();
LANTERN_API void (LANTERN_PTR lantern_randint_like)();
LANTERN_API void (LANTERN_PTR lantern_randint_like)();
LANTERN_API void (LANTERN_PTR lantern_randint_like)();
LANTERN_API void (LANTERN_PTR lantern_randint_like)();
LANTERN_API void (LANTERN_PTR lantern_randn)();
LANTERN_API void (LANTERN_PTR lantern_randn)();
LANTERN_API void (LANTERN_PTR lantern_randn)();
LANTERN_API void (LANTERN_PTR lantern_randn)();
LANTERN_API void (LANTERN_PTR lantern_randn_out)();
LANTERN_API void (LANTERN_PTR lantern_randn_out)();
LANTERN_API void (LANTERN_PTR lantern_randn_like)();
LANTERN_API void (LANTERN_PTR lantern_randn_like)();
LANTERN_API void (LANTERN_PTR lantern_randperm)();
LANTERN_API void (LANTERN_PTR lantern_randperm)();
LANTERN_API void (LANTERN_PTR lantern_randperm_out)();
LANTERN_API void (LANTERN_PTR lantern_randperm_out)();
LANTERN_API void (LANTERN_PTR lantern_range)();
LANTERN_API void (LANTERN_PTR lantern_range)();
LANTERN_API void (LANTERN_PTR lantern_range_out)();
LANTERN_API void (LANTERN_PTR lantern_reciprocal)();
LANTERN_API void (LANTERN_PTR lantern_reciprocal_)();
LANTERN_API void (LANTERN_PTR lantern_reciprocal_out)();
LANTERN_API void (LANTERN_PTR lantern_neg)();
LANTERN_API void (LANTERN_PTR lantern_neg_)();
LANTERN_API void (LANTERN_PTR lantern_neg_out)();
LANTERN_API void (LANTERN_PTR lantern_repeat)();
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)();
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)();
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)();
LANTERN_API void (LANTERN_PTR lantern_reshape)();
LANTERN_API void (LANTERN_PTR lantern__mkldnn_reshape)();
LANTERN_API void (LANTERN_PTR lantern_reshape_as)();
LANTERN_API void (LANTERN_PTR lantern_round)();
LANTERN_API void (LANTERN_PTR lantern_round_)();
LANTERN_API void (LANTERN_PTR lantern_round_out)();
LANTERN_API void (LANTERN_PTR lantern_rrelu)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_)();
LANTERN_API void (LANTERN_PTR lantern_relu)();
LANTERN_API void (LANTERN_PTR lantern_relu_)();
LANTERN_API void (LANTERN_PTR lantern_prelu)();
LANTERN_API void (LANTERN_PTR lantern_prelu_backward)();
LANTERN_API void (LANTERN_PTR lantern_gelu)();
LANTERN_API void (LANTERN_PTR lantern_gelu_backward)();
LANTERN_API void (LANTERN_PTR lantern_hardshrink)();
LANTERN_API void (LANTERN_PTR lantern_hardshrink_backward)();
LANTERN_API void (LANTERN_PTR lantern_rsqrt)();
LANTERN_API void (LANTERN_PTR lantern_rsqrt_)();
LANTERN_API void (LANTERN_PTR lantern_rsqrt_out)();
LANTERN_API void (LANTERN_PTR lantern_select)();
LANTERN_API void (LANTERN_PTR lantern_select)();
LANTERN_API void (LANTERN_PTR lantern_selu)();
LANTERN_API void (LANTERN_PTR lantern_selu_)();
LANTERN_API void (LANTERN_PTR lantern_celu)();
LANTERN_API void (LANTERN_PTR lantern_celu_)();
LANTERN_API void (LANTERN_PTR lantern_sigmoid)();
LANTERN_API void (LANTERN_PTR lantern_sigmoid_)();
LANTERN_API void (LANTERN_PTR lantern_sigmoid_out)();
LANTERN_API void (LANTERN_PTR lantern_sin)();
LANTERN_API void (LANTERN_PTR lantern_sin_)();
LANTERN_API void (LANTERN_PTR lantern_sin_out)();
LANTERN_API void (LANTERN_PTR lantern_sinh)();
LANTERN_API void (LANTERN_PTR lantern_sinh_)();
LANTERN_API void (LANTERN_PTR lantern_sinh_out)();
LANTERN_API void (LANTERN_PTR lantern_detach)();
LANTERN_API void (LANTERN_PTR lantern_detach_)();
LANTERN_API void (LANTERN_PTR lantern_size)();
LANTERN_API void (LANTERN_PTR lantern_size)();
LANTERN_API void (LANTERN_PTR lantern_slice)();
LANTERN_API void (LANTERN_PTR lantern_slogdet)();
LANTERN_API void (LANTERN_PTR lantern_smm)();
LANTERN_API void (LANTERN_PTR lantern_softmax)();
LANTERN_API void (LANTERN_PTR lantern_softmax)();
LANTERN_API void (LANTERN_PTR lantern__softmax)();
LANTERN_API void (LANTERN_PTR lantern__softmax_backward_data)();
LANTERN_API void (LANTERN_PTR lantern_split)();
LANTERN_API void (LANTERN_PTR lantern_split_with_sizes)();
LANTERN_API void (LANTERN_PTR lantern_squeeze)();
LANTERN_API void (LANTERN_PTR lantern_squeeze)();
LANTERN_API void (LANTERN_PTR lantern_squeeze)();
LANTERN_API void (LANTERN_PTR lantern_squeeze_)();
LANTERN_API void (LANTERN_PTR lantern_squeeze_)();
LANTERN_API void (LANTERN_PTR lantern_squeeze_)();
LANTERN_API void (LANTERN_PTR lantern_sspaddmm)();
LANTERN_API void (LANTERN_PTR lantern_sspaddmm_out)();
LANTERN_API void (LANTERN_PTR lantern_stack)();
LANTERN_API void (LANTERN_PTR lantern_stack_out)();
LANTERN_API void (LANTERN_PTR lantern_stft)();
LANTERN_API void (LANTERN_PTR lantern_stride)();
LANTERN_API void (LANTERN_PTR lantern_stride)();
LANTERN_API void (LANTERN_PTR lantern_sum)();
LANTERN_API void (LANTERN_PTR lantern_sum)();
LANTERN_API void (LANTERN_PTR lantern_sum)();
LANTERN_API void (LANTERN_PTR lantern_sum_out)();
LANTERN_API void (LANTERN_PTR lantern_sum_out)();
LANTERN_API void (LANTERN_PTR lantern_sum_to_size)();
LANTERN_API void (LANTERN_PTR lantern_sqrt)();
LANTERN_API void (LANTERN_PTR lantern_sqrt_)();
LANTERN_API void (LANTERN_PTR lantern_sqrt_out)();
LANTERN_API void (LANTERN_PTR lantern_std)();
LANTERN_API void (LANTERN_PTR lantern_std)();
LANTERN_API void (LANTERN_PTR lantern_std_mean)();
LANTERN_API void (LANTERN_PTR lantern_std_mean)();
LANTERN_API void (LANTERN_PTR lantern_std_mean)();
LANTERN_API void (LANTERN_PTR lantern_std_out)();
LANTERN_API void (LANTERN_PTR lantern_std)();
LANTERN_API void (LANTERN_PTR lantern_std_out)();
LANTERN_API void (LANTERN_PTR lantern_prod)();
LANTERN_API void (LANTERN_PTR lantern_prod)();
LANTERN_API void (LANTERN_PTR lantern_prod_out)();
LANTERN_API void (LANTERN_PTR lantern_prod)();
LANTERN_API void (LANTERN_PTR lantern_prod_out)();
LANTERN_API void (LANTERN_PTR lantern_t)();
LANTERN_API void (LANTERN_PTR lantern_t_)();
LANTERN_API void (LANTERN_PTR lantern_tan)();
LANTERN_API void (LANTERN_PTR lantern_tan_)();
LANTERN_API void (LANTERN_PTR lantern_tan_out)();
LANTERN_API void (LANTERN_PTR lantern_tanh)();
LANTERN_API void (LANTERN_PTR lantern_tanh_)();
LANTERN_API void (LANTERN_PTR lantern_tanh_out)();
LANTERN_API void (LANTERN_PTR lantern_tensordot)();
LANTERN_API void (LANTERN_PTR lantern_threshold)();
LANTERN_API void (LANTERN_PTR lantern_threshold_)();
LANTERN_API void (LANTERN_PTR lantern_threshold_out)();
LANTERN_API void (LANTERN_PTR lantern_threshold_backward)();
LANTERN_API void (LANTERN_PTR lantern_transpose)();
LANTERN_API void (LANTERN_PTR lantern_transpose)();
LANTERN_API void (LANTERN_PTR lantern__mkldnn_transpose)();
LANTERN_API void (LANTERN_PTR lantern_transpose_)();
LANTERN_API void (LANTERN_PTR lantern__mkldnn_transpose_)();
LANTERN_API void (LANTERN_PTR lantern_one_hot)();
LANTERN_API void (LANTERN_PTR lantern_flip)();
LANTERN_API void (LANTERN_PTR lantern_roll)();
LANTERN_API void (LANTERN_PTR lantern_rot90)();
LANTERN_API void (LANTERN_PTR lantern_trapz)();
LANTERN_API void (LANTERN_PTR lantern_trapz)();
LANTERN_API void (LANTERN_PTR lantern__trilinear)();
LANTERN_API void (LANTERN_PTR lantern_triplet_margin_loss)();
LANTERN_API void (LANTERN_PTR lantern_trunc)();
LANTERN_API void (LANTERN_PTR lantern_trunc_)();
LANTERN_API void (LANTERN_PTR lantern_trunc_out)();
LANTERN_API void (LANTERN_PTR lantern_type_as)();
LANTERN_API void (LANTERN_PTR lantern__has_compatible_shallow_copy_type)();
LANTERN_API void (LANTERN_PTR lantern__unique)();
LANTERN_API void (LANTERN_PTR lantern_unique_dim)();
LANTERN_API void (LANTERN_PTR lantern_unique_consecutive)();
LANTERN_API void (LANTERN_PTR lantern_unique_dim_consecutive)();
LANTERN_API void (LANTERN_PTR lantern__unique2)();
LANTERN_API void (LANTERN_PTR lantern__unsafe_view)();
LANTERN_API void (LANTERN_PTR lantern_unsqueeze)();
LANTERN_API void (LANTERN_PTR lantern_unsqueeze_)();
LANTERN_API void (LANTERN_PTR lantern_var)();
LANTERN_API void (LANTERN_PTR lantern_var)();
LANTERN_API void (LANTERN_PTR lantern_var_out)();
LANTERN_API void (LANTERN_PTR lantern_var)();
LANTERN_API void (LANTERN_PTR lantern_var_out)();
LANTERN_API void (LANTERN_PTR lantern_var_mean)();
LANTERN_API void (LANTERN_PTR lantern_var_mean)();
LANTERN_API void (LANTERN_PTR lantern_var_mean)();
LANTERN_API void (LANTERN_PTR lantern_view_as)();
LANTERN_API void (LANTERN_PTR lantern_where)();
LANTERN_API void (LANTERN_PTR lantern_where)();
LANTERN_API void (LANTERN_PTR lantern__s_where)();
LANTERN_API void (LANTERN_PTR lantern_norm_except_dim)();
LANTERN_API void (LANTERN_PTR lantern__weight_norm)();
LANTERN_API void (LANTERN_PTR lantern__weight_norm_cuda_interface)();
LANTERN_API void (LANTERN_PTR lantern__weight_norm_cuda_interface_backward)();
LANTERN_API void (LANTERN_PTR lantern__weight_norm_differentiable_backward)();
LANTERN_API void (LANTERN_PTR lantern_zeros)();
LANTERN_API void (LANTERN_PTR lantern_zeros)();
LANTERN_API void (LANTERN_PTR lantern_zeros_out)();
LANTERN_API void (LANTERN_PTR lantern_zeros_like)();
LANTERN_API void (LANTERN_PTR lantern_zeros_like)();
LANTERN_API void (LANTERN_PTR lantern__standard_gamma_grad)();
LANTERN_API void (LANTERN_PTR lantern__standard_gamma)();
LANTERN_API void (LANTERN_PTR lantern__dirichlet_grad)();
LANTERN_API void (LANTERN_PTR lantern__sample_dirichlet)();
LANTERN_API void (LANTERN_PTR lantern_poisson)();
LANTERN_API void (LANTERN_PTR lantern_native_norm)();
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)();
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)();
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)();
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)();
LANTERN_API void (LANTERN_PTR lantern__sparse_sum_backward)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm)();
LANTERN_API void (LANTERN_PTR lantern_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm)();
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm)();
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm)();
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm)();
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm_out)();
LANTERN_API void (LANTERN_PTR lantern_clone)();
LANTERN_API void (LANTERN_PTR lantern_resize_as_)();
LANTERN_API void (LANTERN_PTR lantern_pow_out)();
LANTERN_API void (LANTERN_PTR lantern_pow)();
LANTERN_API void (LANTERN_PTR lantern_zero_)();
LANTERN_API void (LANTERN_PTR lantern_sub_out)();
LANTERN_API void (LANTERN_PTR lantern_sub)();
LANTERN_API void (LANTERN_PTR lantern_sub_)();
LANTERN_API void (LANTERN_PTR lantern_sub)();
LANTERN_API void (LANTERN_PTR lantern_sub_)();
LANTERN_API void (LANTERN_PTR lantern_rsub)();
LANTERN_API void (LANTERN_PTR lantern_rsub)();
LANTERN_API void (LANTERN_PTR lantern__sparse_addmm)();
LANTERN_API void (LANTERN_PTR lantern_addmm_out)();
LANTERN_API void (LANTERN_PTR lantern_addmm)();
LANTERN_API void (LANTERN_PTR lantern_addmm_)();
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_unsafe)();
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_with_dims)();
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_with_dims_and_tensors)();
LANTERN_API void (LANTERN_PTR lantern_sparse_resize_)();
LANTERN_API void (LANTERN_PTR lantern_sparse_resize_and_clear_)();
LANTERN_API void (LANTERN_PTR lantern_sparse_mask)();
LANTERN_API void (LANTERN_PTR lantern_to_dense)();
LANTERN_API void (LANTERN_PTR lantern_to_dense_backward)();
LANTERN_API void (LANTERN_PTR lantern_sparse_dim)();
LANTERN_API void (LANTERN_PTR lantern__dimi)();
LANTERN_API void (LANTERN_PTR lantern_dense_dim)();
LANTERN_API void (LANTERN_PTR lantern__dimv)();
LANTERN_API void (LANTERN_PTR lantern__nnz)();
LANTERN_API void (LANTERN_PTR lantern_coalesce)();
LANTERN_API void (LANTERN_PTR lantern_is_coalesced)();
LANTERN_API void (LANTERN_PTR lantern__indices)();
LANTERN_API void (LANTERN_PTR lantern__values)();
LANTERN_API void (LANTERN_PTR lantern__coalesced_)();
LANTERN_API void (LANTERN_PTR lantern_indices)();
LANTERN_API void (LANTERN_PTR lantern_values)();
LANTERN_API void (LANTERN_PTR lantern_hspmm_out)();
LANTERN_API void (LANTERN_PTR lantern_hspmm)();
LANTERN_API void (LANTERN_PTR lantern_copy_sparse_to_sparse_)();
LANTERN_API void (LANTERN_PTR lantern_numel)();
LANTERN_API void (LANTERN_PTR lantern_unbind)();
LANTERN_API void (LANTERN_PTR lantern_unbind)();
LANTERN_API void (LANTERN_PTR lantern_to_sparse)();
LANTERN_API void (LANTERN_PTR lantern_to_sparse)();
LANTERN_API void (LANTERN_PTR lantern_to_mkldnn)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_reorder_conv2d_weight)();
LANTERN_API void (LANTERN_PTR lantern_to_mkldnn_backward)();
LANTERN_API void (LANTERN_PTR lantern_quantize_per_tensor)();
LANTERN_API void (LANTERN_PTR lantern_quantize_per_channel)();
LANTERN_API void (LANTERN_PTR lantern_dequantize)();
LANTERN_API void (LANTERN_PTR lantern_q_scale)();
LANTERN_API void (LANTERN_PTR lantern_q_zero_point)();
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_scales)();
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_zero_points)();
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_axis)();
LANTERN_API void (LANTERN_PTR lantern_int_repr)();
LANTERN_API void (LANTERN_PTR lantern__make_per_tensor_quantized_tensor)();
LANTERN_API void (LANTERN_PTR lantern__make_per_channel_quantized_tensor)();
LANTERN_API void (LANTERN_PTR lantern_qscheme)();
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_tensor_affine)();
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_tensor_affine_backward)();
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_channel_affine)();
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_channel_affine_backward)();
LANTERN_API void (LANTERN_PTR lantern_to)();
LANTERN_API void (LANTERN_PTR lantern_to)();
LANTERN_API void (LANTERN_PTR lantern_to)();
LANTERN_API void (LANTERN_PTR lantern_to)();
LANTERN_API void (LANTERN_PTR lantern_meshgrid)();
LANTERN_API void (LANTERN_PTR lantern_cartesian_prod)();
LANTERN_API void (LANTERN_PTR lantern_combinations)();
LANTERN_API void (LANTERN_PTR lantern_item)();
LANTERN_API void (LANTERN_PTR lantern_result_type)();
LANTERN_API void (LANTERN_PTR lantern_result_type)();
LANTERN_API void (LANTERN_PTR lantern_result_type)();
LANTERN_API void (LANTERN_PTR lantern_result_type)();
LANTERN_API void (LANTERN_PTR lantern_can_cast)();
LANTERN_API void (LANTERN_PTR lantern_promote_types)();
LANTERN_API void (LANTERN_PTR lantern__local_scalar_dense)();
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_lstm_cell)();
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_lstm_cell_backward)();
LANTERN_API void (LANTERN_PTR lantern__thnn_differentiable_lstm_cell_backward)();
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_gru_cell)();
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_gru_cell_backward)();
LANTERN_API void (LANTERN_PTR lantern__thnn_differentiable_gru_cell_backward)();
LANTERN_API void (LANTERN_PTR lantern_lstm)();
LANTERN_API void (LANTERN_PTR lantern_lstm)();
LANTERN_API void (LANTERN_PTR lantern_gru)();
LANTERN_API void (LANTERN_PTR lantern_gru)();
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh)();
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh)();
LANTERN_API void (LANTERN_PTR lantern_rnn_relu)();
LANTERN_API void (LANTERN_PTR lantern_rnn_relu)();
LANTERN_API void (LANTERN_PTR lantern_lstm_cell)();
LANTERN_API void (LANTERN_PTR lantern_gru_cell)();
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh_cell)();
LANTERN_API void (LANTERN_PTR lantern_rnn_relu_cell)();
LANTERN_API void (LANTERN_PTR lantern_quantized_lstm)();
LANTERN_API void (LANTERN_PTR lantern_quantized_gru)();
LANTERN_API void (LANTERN_PTR lantern_quantized_gru)();
LANTERN_API void (LANTERN_PTR lantern_quantized_lstm_cell)();
LANTERN_API void (LANTERN_PTR lantern_quantized_gru_cell)();
LANTERN_API void (LANTERN_PTR lantern_quantized_rnn_relu_cell)();
LANTERN_API void (LANTERN_PTR lantern_quantized_rnn_tanh_cell)();
LANTERN_API void (LANTERN_PTR lantern__pack_padded_sequence)();
LANTERN_API void (LANTERN_PTR lantern__pack_padded_sequence_backward)();
LANTERN_API void (LANTERN_PTR lantern__pad_packed_sequence)();
LANTERN_API void (LANTERN_PTR lantern_set_)();
LANTERN_API void (LANTERN_PTR lantern_set_)();
LANTERN_API void (LANTERN_PTR lantern_set_)();
LANTERN_API void (LANTERN_PTR lantern_set_)();
LANTERN_API void (LANTERN_PTR lantern_set_quantizer_)();
LANTERN_API void (LANTERN_PTR lantern_is_set_to)();
LANTERN_API void (LANTERN_PTR lantern_masked_fill_)();
LANTERN_API void (LANTERN_PTR lantern_masked_fill)();
LANTERN_API void (LANTERN_PTR lantern_masked_fill_)();
LANTERN_API void (LANTERN_PTR lantern_masked_fill)();
LANTERN_API void (LANTERN_PTR lantern_masked_scatter_)();
LANTERN_API void (LANTERN_PTR lantern_masked_scatter)();
LANTERN_API void (LANTERN_PTR lantern_view)();
LANTERN_API void (LANTERN_PTR lantern_put_)();
LANTERN_API void (LANTERN_PTR lantern_index_add_)();
LANTERN_API void (LANTERN_PTR lantern_index_add)();
LANTERN_API void (LANTERN_PTR lantern_index_add)();
LANTERN_API void (LANTERN_PTR lantern_index_fill_)();
LANTERN_API void (LANTERN_PTR lantern_index_fill)();
LANTERN_API void (LANTERN_PTR lantern_index_fill_)();
LANTERN_API void (LANTERN_PTR lantern_index_fill)();
LANTERN_API void (LANTERN_PTR lantern_index_fill_)();
LANTERN_API void (LANTERN_PTR lantern_index_fill_)();
LANTERN_API void (LANTERN_PTR lantern_index_fill)();
LANTERN_API void (LANTERN_PTR lantern_index_fill)();
LANTERN_API void (LANTERN_PTR lantern_scatter_)();
LANTERN_API void (LANTERN_PTR lantern_scatter)();
LANTERN_API void (LANTERN_PTR lantern_scatter_)();
LANTERN_API void (LANTERN_PTR lantern_scatter)();
LANTERN_API void (LANTERN_PTR lantern_scatter)();
LANTERN_API void (LANTERN_PTR lantern_scatter)();
LANTERN_API void (LANTERN_PTR lantern_scatter_add_)();
LANTERN_API void (LANTERN_PTR lantern_scatter_add)();
LANTERN_API void (LANTERN_PTR lantern_scatter_add)();
LANTERN_API void (LANTERN_PTR lantern_lt_)();
LANTERN_API void (LANTERN_PTR lantern_lt_)();
LANTERN_API void (LANTERN_PTR lantern_gt_)();
LANTERN_API void (LANTERN_PTR lantern_gt_)();
LANTERN_API void (LANTERN_PTR lantern_le_)();
LANTERN_API void (LANTERN_PTR lantern_le_)();
LANTERN_API void (LANTERN_PTR lantern_ge_)();
LANTERN_API void (LANTERN_PTR lantern_ge_)();
LANTERN_API void (LANTERN_PTR lantern_eq_)();
LANTERN_API void (LANTERN_PTR lantern_eq_)();
LANTERN_API void (LANTERN_PTR lantern_ne_)();
LANTERN_API void (LANTERN_PTR lantern_ne_)();
LANTERN_API void (LANTERN_PTR lantern___and__)();
LANTERN_API void (LANTERN_PTR lantern___and__)();
LANTERN_API void (LANTERN_PTR lantern___iand__)();
LANTERN_API void (LANTERN_PTR lantern___iand__)();
LANTERN_API void (LANTERN_PTR lantern___or__)();
LANTERN_API void (LANTERN_PTR lantern___or__)();
LANTERN_API void (LANTERN_PTR lantern___ior__)();
LANTERN_API void (LANTERN_PTR lantern___ior__)();
LANTERN_API void (LANTERN_PTR lantern___xor__)();
LANTERN_API void (LANTERN_PTR lantern___xor__)();
LANTERN_API void (LANTERN_PTR lantern___ixor__)();
LANTERN_API void (LANTERN_PTR lantern___ixor__)();
LANTERN_API void (LANTERN_PTR lantern___lshift__)();
LANTERN_API void (LANTERN_PTR lantern___lshift__)();
LANTERN_API void (LANTERN_PTR lantern___ilshift__)();
LANTERN_API void (LANTERN_PTR lantern___ilshift__)();
LANTERN_API void (LANTERN_PTR lantern___rshift__)();
LANTERN_API void (LANTERN_PTR lantern___rshift__)();
LANTERN_API void (LANTERN_PTR lantern___irshift__)();
LANTERN_API void (LANTERN_PTR lantern___irshift__)();
LANTERN_API void (LANTERN_PTR lantern_lgamma_)();
LANTERN_API void (LANTERN_PTR lantern_atan2_)();
LANTERN_API void (LANTERN_PTR lantern_tril_)();
LANTERN_API void (LANTERN_PTR lantern_triu_)();
LANTERN_API void (LANTERN_PTR lantern_digamma_)();
LANTERN_API void (LANTERN_PTR lantern_polygamma_)();
LANTERN_API void (LANTERN_PTR lantern_renorm_)();
LANTERN_API void (LANTERN_PTR lantern_pow_)();
LANTERN_API void (LANTERN_PTR lantern_pow_)();
LANTERN_API void (LANTERN_PTR lantern_lerp_)();
LANTERN_API void (LANTERN_PTR lantern_lerp_)();
LANTERN_API void (LANTERN_PTR lantern_fmod_)();
LANTERN_API void (LANTERN_PTR lantern_fmod_)();
LANTERN_API void (LANTERN_PTR lantern_remainder_)();
LANTERN_API void (LANTERN_PTR lantern_remainder_)();
LANTERN_API void (LANTERN_PTR lantern_addbmm_)();
LANTERN_API void (LANTERN_PTR lantern_addbmm_out)();
LANTERN_API void (LANTERN_PTR lantern_addbmm)();
LANTERN_API void (LANTERN_PTR lantern_addcdiv_)();
LANTERN_API void (LANTERN_PTR lantern_random_)();
LANTERN_API void (LANTERN_PTR lantern_random_)();
LANTERN_API void (LANTERN_PTR lantern_random_)();
LANTERN_API void (LANTERN_PTR lantern_uniform_)();
LANTERN_API void (LANTERN_PTR lantern_normal_)();
LANTERN_API void (LANTERN_PTR lantern_cauchy_)();
LANTERN_API void (LANTERN_PTR lantern_log_normal_)();
LANTERN_API void (LANTERN_PTR lantern_exponential_)();
LANTERN_API void (LANTERN_PTR lantern_geometric_)();
LANTERN_API void (LANTERN_PTR lantern_diag_out)();
LANTERN_API void (LANTERN_PTR lantern_diag)();
LANTERN_API void (LANTERN_PTR lantern_cross_out)();
LANTERN_API void (LANTERN_PTR lantern_cross)();
LANTERN_API void (LANTERN_PTR lantern_triu_out)();
LANTERN_API void (LANTERN_PTR lantern_triu)();
LANTERN_API void (LANTERN_PTR lantern_tril_out)();
LANTERN_API void (LANTERN_PTR lantern_tril)();
LANTERN_API void (LANTERN_PTR lantern_tril_indices)();
LANTERN_API void (LANTERN_PTR lantern_triu_indices)();
LANTERN_API void (LANTERN_PTR lantern_trace)();
LANTERN_API void (LANTERN_PTR lantern_ne_out)();
LANTERN_API void (LANTERN_PTR lantern_ne)();
LANTERN_API void (LANTERN_PTR lantern_ne_out)();
LANTERN_API void (LANTERN_PTR lantern_ne)();
LANTERN_API void (LANTERN_PTR lantern_eq_out)();
LANTERN_API void (LANTERN_PTR lantern_eq)();
LANTERN_API void (LANTERN_PTR lantern_eq_out)();
LANTERN_API void (LANTERN_PTR lantern_eq)();
LANTERN_API void (LANTERN_PTR lantern_ge_out)();
LANTERN_API void (LANTERN_PTR lantern_ge)();
LANTERN_API void (LANTERN_PTR lantern_ge_out)();
LANTERN_API void (LANTERN_PTR lantern_ge)();
LANTERN_API void (LANTERN_PTR lantern_le_out)();
LANTERN_API void (LANTERN_PTR lantern_le)();
LANTERN_API void (LANTERN_PTR lantern_le_out)();
LANTERN_API void (LANTERN_PTR lantern_le)();
LANTERN_API void (LANTERN_PTR lantern_gt_out)();
LANTERN_API void (LANTERN_PTR lantern_gt)();
LANTERN_API void (LANTERN_PTR lantern_gt_out)();
LANTERN_API void (LANTERN_PTR lantern_gt)();
LANTERN_API void (LANTERN_PTR lantern_lt_out)();
LANTERN_API void (LANTERN_PTR lantern_lt)();
LANTERN_API void (LANTERN_PTR lantern_lt_out)();
LANTERN_API void (LANTERN_PTR lantern_lt)();
LANTERN_API void (LANTERN_PTR lantern_take_out)();
LANTERN_API void (LANTERN_PTR lantern_take)();
LANTERN_API void (LANTERN_PTR lantern_index_select_out)();
LANTERN_API void (LANTERN_PTR lantern_index_select)();
LANTERN_API void (LANTERN_PTR lantern_index_select_out)();
LANTERN_API void (LANTERN_PTR lantern_index_select)();
LANTERN_API void (LANTERN_PTR lantern_masked_select_out)();
LANTERN_API void (LANTERN_PTR lantern_masked_select)();
LANTERN_API void (LANTERN_PTR lantern_nonzero_out)();
LANTERN_API void (LANTERN_PTR lantern_nonzero)();
LANTERN_API void (LANTERN_PTR lantern_nonzero_numpy)();
LANTERN_API void (LANTERN_PTR lantern_gather_out)();
LANTERN_API void (LANTERN_PTR lantern_gather)();
LANTERN_API void (LANTERN_PTR lantern_gather_out)();
LANTERN_API void (LANTERN_PTR lantern_gather)();
LANTERN_API void (LANTERN_PTR lantern__gather_sparse_backward)();
LANTERN_API void (LANTERN_PTR lantern_addcmul_out)();
LANTERN_API void (LANTERN_PTR lantern_addcmul)();
LANTERN_API void (LANTERN_PTR lantern_addcmul_)();
LANTERN_API void (LANTERN_PTR lantern_addcdiv_out)();
LANTERN_API void (LANTERN_PTR lantern_addcdiv)();
LANTERN_API void (LANTERN_PTR lantern_lstsq_out)();
LANTERN_API void (LANTERN_PTR lantern_lstsq)();
LANTERN_API void (LANTERN_PTR lantern_triangular_solve_out)();
LANTERN_API void (LANTERN_PTR lantern_triangular_solve)();
LANTERN_API void (LANTERN_PTR lantern__triangular_solve_helper)();
LANTERN_API void (LANTERN_PTR lantern_symeig_out)();
LANTERN_API void (LANTERN_PTR lantern_symeig)();
LANTERN_API void (LANTERN_PTR lantern__symeig_helper)();
LANTERN_API void (LANTERN_PTR lantern_eig_out)();
LANTERN_API void (LANTERN_PTR lantern_eig)();
LANTERN_API void (LANTERN_PTR lantern_svd_out)();
LANTERN_API void (LANTERN_PTR lantern_svd)();
LANTERN_API void (LANTERN_PTR lantern__svd_helper)();
LANTERN_API void (LANTERN_PTR lantern_cholesky_out)();
LANTERN_API void (LANTERN_PTR lantern_cholesky)();
LANTERN_API void (LANTERN_PTR lantern__cholesky_helper)();
LANTERN_API void (LANTERN_PTR lantern_cholesky_solve_out)();
LANTERN_API void (LANTERN_PTR lantern_cholesky_solve)();
LANTERN_API void (LANTERN_PTR lantern__cholesky_solve_helper)();
LANTERN_API void (LANTERN_PTR lantern_solve)();
LANTERN_API void (LANTERN_PTR lantern_solve_out)();
LANTERN_API void (LANTERN_PTR lantern__solve_helper)();
LANTERN_API void (LANTERN_PTR lantern_cholesky_inverse_out)();
LANTERN_API void (LANTERN_PTR lantern_cholesky_inverse)();
LANTERN_API void (LANTERN_PTR lantern_qr_out)();
LANTERN_API void (LANTERN_PTR lantern_qr)();
LANTERN_API void (LANTERN_PTR lantern__qr_helper)();
LANTERN_API void (LANTERN_PTR lantern_geqrf_out)();
LANTERN_API void (LANTERN_PTR lantern_geqrf)();
LANTERN_API void (LANTERN_PTR lantern_orgqr_out)();
LANTERN_API void (LANTERN_PTR lantern_orgqr)();
LANTERN_API void (LANTERN_PTR lantern_ormqr_out)();
LANTERN_API void (LANTERN_PTR lantern_ormqr)();
LANTERN_API void (LANTERN_PTR lantern__lu_with_info)();
LANTERN_API void (LANTERN_PTR lantern_lu_solve_out)();
LANTERN_API void (LANTERN_PTR lantern_lu_solve)();
LANTERN_API void (LANTERN_PTR lantern__lu_solve_helper)();
LANTERN_API void (LANTERN_PTR lantern_multinomial_out)();
LANTERN_API void (LANTERN_PTR lantern_multinomial)();
LANTERN_API void (LANTERN_PTR lantern__multinomial_alias_setup)();
LANTERN_API void (LANTERN_PTR lantern__multinomial_alias_draw)();
LANTERN_API void (LANTERN_PTR lantern_lgamma_out)();
LANTERN_API void (LANTERN_PTR lantern_lgamma)();
LANTERN_API void (LANTERN_PTR lantern_digamma_out)();
LANTERN_API void (LANTERN_PTR lantern_digamma)();
LANTERN_API void (LANTERN_PTR lantern_polygamma_out)();
LANTERN_API void (LANTERN_PTR lantern_polygamma)();
LANTERN_API void (LANTERN_PTR lantern_erfinv)();
LANTERN_API void (LANTERN_PTR lantern_erfinv_)();
LANTERN_API void (LANTERN_PTR lantern_erfinv_out)();
LANTERN_API void (LANTERN_PTR lantern_sign)();
LANTERN_API void (LANTERN_PTR lantern_sign_)();
LANTERN_API void (LANTERN_PTR lantern_sign_out)();
LANTERN_API void (LANTERN_PTR lantern_dist)();
LANTERN_API void (LANTERN_PTR lantern_atan2_out)();
LANTERN_API void (LANTERN_PTR lantern_atan2)();
LANTERN_API void (LANTERN_PTR lantern_lerp_out)();
LANTERN_API void (LANTERN_PTR lantern_lerp_out)();
LANTERN_API void (LANTERN_PTR lantern_lerp)();
LANTERN_API void (LANTERN_PTR lantern_lerp)();
LANTERN_API void (LANTERN_PTR lantern_histc_out)();
LANTERN_API void (LANTERN_PTR lantern_histc)();
LANTERN_API void (LANTERN_PTR lantern_fmod_out)();
LANTERN_API void (LANTERN_PTR lantern_fmod)();
LANTERN_API void (LANTERN_PTR lantern_fmod_out)();
LANTERN_API void (LANTERN_PTR lantern_fmod)();
LANTERN_API void (LANTERN_PTR lantern_remainder_out)();
LANTERN_API void (LANTERN_PTR lantern_remainder)();
LANTERN_API void (LANTERN_PTR lantern_remainder_out)();
LANTERN_API void (LANTERN_PTR lantern_remainder)();
LANTERN_API void (LANTERN_PTR lantern_min_out)();
LANTERN_API void (LANTERN_PTR lantern_min)();
LANTERN_API void (LANTERN_PTR lantern_min)();
LANTERN_API void (LANTERN_PTR lantern_max_out)();
LANTERN_API void (LANTERN_PTR lantern_max)();
LANTERN_API void (LANTERN_PTR lantern_max)();
LANTERN_API void (LANTERN_PTR lantern_median)();
LANTERN_API void (LANTERN_PTR lantern_sort_out)();
LANTERN_API void (LANTERN_PTR lantern_sort)();
LANTERN_API void (LANTERN_PTR lantern_sort_out)();
LANTERN_API void (LANTERN_PTR lantern_sort)();
LANTERN_API void (LANTERN_PTR lantern_argsort)();
LANTERN_API void (LANTERN_PTR lantern_argsort)();
LANTERN_API void (LANTERN_PTR lantern_topk_out)();
LANTERN_API void (LANTERN_PTR lantern_topk)();
LANTERN_API void (LANTERN_PTR lantern_all)();
LANTERN_API void (LANTERN_PTR lantern_any)();
LANTERN_API void (LANTERN_PTR lantern_renorm_out)();
LANTERN_API void (LANTERN_PTR lantern_renorm)();
LANTERN_API void (LANTERN_PTR lantern_unfold)();
LANTERN_API void (LANTERN_PTR lantern_equal)();
LANTERN_API void (LANTERN_PTR lantern_pow_out)();
LANTERN_API void (LANTERN_PTR lantern_pow)();
LANTERN_API void (LANTERN_PTR lantern_pow_out)();
LANTERN_API void (LANTERN_PTR lantern_pow)();
LANTERN_API void (LANTERN_PTR lantern_normal_out)();
LANTERN_API void (LANTERN_PTR lantern_normal)();
LANTERN_API void (LANTERN_PTR lantern_normal_out)();
LANTERN_API void (LANTERN_PTR lantern_normal)();
LANTERN_API void (LANTERN_PTR lantern_normal_out)();
LANTERN_API void (LANTERN_PTR lantern_normal)();
LANTERN_API void (LANTERN_PTR lantern_normal)();
LANTERN_API void (LANTERN_PTR lantern_normal_out)();
LANTERN_API void (LANTERN_PTR lantern_alias)();
LANTERN_API void (LANTERN_PTR lantern__addr)();
LANTERN_API void (LANTERN_PTR lantern__addr_)();
LANTERN_API void (LANTERN_PTR lantern__addr_out)();
LANTERN_API void (LANTERN_PTR lantern__index_copy_)();
LANTERN_API void (LANTERN_PTR lantern__cumsum)();
LANTERN_API void (LANTERN_PTR lantern__cumsum_out)();
LANTERN_API void (LANTERN_PTR lantern__cumprod)();
LANTERN_API void (LANTERN_PTR lantern__cumprod_out)();
LANTERN_API void (LANTERN_PTR lantern__var)();
LANTERN_API void (LANTERN_PTR lantern__std)();
LANTERN_API void (LANTERN_PTR lantern__cat)();
LANTERN_API void (LANTERN_PTR lantern__cat_out)();
LANTERN_API void (LANTERN_PTR lantern__mode)();
LANTERN_API void (LANTERN_PTR lantern__mode_out)();
LANTERN_API void (LANTERN_PTR lantern__max)();
LANTERN_API void (LANTERN_PTR lantern__max_out)();
LANTERN_API void (LANTERN_PTR lantern__min)();
LANTERN_API void (LANTERN_PTR lantern__min_out)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_out)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_backward)();
LANTERN_API void (LANTERN_PTR lantern_mse_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_mse_loss)();
LANTERN_API void (LANTERN_PTR lantern_mse_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_mse_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_l1_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_l1_loss)();
LANTERN_API void (LANTERN_PTR lantern_l1_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_l1_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss)();
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_forward)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss_forward)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_forward)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss)();
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_out)();
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss)();
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR lantern_elu_out)();
LANTERN_API void (LANTERN_PTR lantern_elu)();
LANTERN_API void (LANTERN_PTR lantern_elu_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_elu_backward)();
LANTERN_API void (LANTERN_PTR lantern_elu_)();
LANTERN_API void (LANTERN_PTR lantern_glu_out)();
LANTERN_API void (LANTERN_PTR lantern_glu)();
LANTERN_API void (LANTERN_PTR lantern_glu_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_glu_backward)();
LANTERN_API void (LANTERN_PTR lantern_hardtanh_out)();
LANTERN_API void (LANTERN_PTR lantern_hardtanh)();
LANTERN_API void (LANTERN_PTR lantern_hardtanh_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_hardtanh_backward)();
LANTERN_API void (LANTERN_PTR lantern_hardtanh_)();
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_out)();
LANTERN_API void (LANTERN_PTR lantern_leaky_relu)();
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_backward)();
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_out)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_forward)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_backward)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_out)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_backward)();
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_)();
LANTERN_API void (LANTERN_PTR lantern_softplus_out)();
LANTERN_API void (LANTERN_PTR lantern_softplus)();
LANTERN_API void (LANTERN_PTR lantern_softplus_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_softplus_backward)();
LANTERN_API void (LANTERN_PTR lantern_softshrink_out)();
LANTERN_API void (LANTERN_PTR lantern_softshrink)();
LANTERN_API void (LANTERN_PTR lantern_softshrink_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_softshrink_backward)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool2d_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_mkldnn_adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR lantern__adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR lantern__adaptive_avg_pool2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_out)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_out)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_out)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_out)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_out)();
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices)();
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_backward)();
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_out)();
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices)();
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_backward)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_out)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_out)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_out)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_backward)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_out)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_backward)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_sigmoid_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_sigmoid_backward)();
LANTERN_API void (LANTERN_PTR lantern_tanh_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_tanh_backward)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_out)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_out)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_forward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_forward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_forward)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated2d)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated2d_backward)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated3d)();
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated3d_backward)();
LANTERN_API void (LANTERN_PTR lantern_col2im_out)();
LANTERN_API void (LANTERN_PTR lantern_col2im)();
LANTERN_API void (LANTERN_PTR lantern_col2im_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_col2im_backward)();
LANTERN_API void (LANTERN_PTR lantern_im2col_out)();
LANTERN_API void (LANTERN_PTR lantern_im2col)();
LANTERN_API void (LANTERN_PTR lantern_im2col_backward_out)();
LANTERN_API void (LANTERN_PTR lantern_im2col_backward)();
*/
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
  /*
  LOAD_SYMBOL(lantern__cast_byte)
  LOAD_SYMBOL(lantern__cast_char)
  LOAD_SYMBOL(lantern__cast_double)
  LOAD_SYMBOL(lantern__cast_float)
  LOAD_SYMBOL(lantern__cast_int)
  LOAD_SYMBOL(lantern__cast_long)
  LOAD_SYMBOL(lantern__cast_short)
  LOAD_SYMBOL(lantern__cast_half)
  LOAD_SYMBOL(lantern_backward)
  LOAD_SYMBOL(lantern_set_data)
  LOAD_SYMBOL(lantern_data)
  LOAD_SYMBOL(lantern_is_leaf)
  LOAD_SYMBOL(lantern_output_nr)
  LOAD_SYMBOL(lantern__version)
  LOAD_SYMBOL(lantern_rename_)
  LOAD_SYMBOL(lantern_rename)
  LOAD_SYMBOL(lantern_align_to)
  LOAD_SYMBOL(lantern_align_as)
  LOAD_SYMBOL(lantern_align_tensors)
  LOAD_SYMBOL(lantern_refine_names)
  LOAD_SYMBOL(lantern_unflatten)
  LOAD_SYMBOL(lantern_unflatten)
  LOAD_SYMBOL(lantern__cudnn_ctc_loss)
  LOAD_SYMBOL(lantern__cudnn_rnn_flatten_weight)
  LOAD_SYMBOL(lantern__cudnn_rnn)
  LOAD_SYMBOL(lantern__cudnn_rnn_backward)
  LOAD_SYMBOL(lantern__cudnn_init_dropout_state)
  LOAD_SYMBOL(lantern__debug_has_internal_overlap)
  LOAD_SYMBOL(lantern__fused_dropout)
  LOAD_SYMBOL(lantern__masked_scale)
  LOAD_SYMBOL(lantern__sobol_engine_draw)
  LOAD_SYMBOL(lantern__sobol_engine_ff_)
  LOAD_SYMBOL(lantern__sobol_engine_scramble_)
  LOAD_SYMBOL(lantern__sobol_engine_initialize_state_)
  LOAD_SYMBOL(lantern__reshape_from_tensor)
  LOAD_SYMBOL(lantern__shape_as_tensor)
  LOAD_SYMBOL(lantern_dropout)
  LOAD_SYMBOL(lantern_dropout_)
  LOAD_SYMBOL(lantern_feature_dropout)
  LOAD_SYMBOL(lantern_feature_dropout_)
  LOAD_SYMBOL(lantern_alpha_dropout)
  LOAD_SYMBOL(lantern_alpha_dropout_)
  LOAD_SYMBOL(lantern_feature_alpha_dropout)
  LOAD_SYMBOL(lantern_feature_alpha_dropout_)
  LOAD_SYMBOL(lantern_abs)
  LOAD_SYMBOL(lantern_abs_)
  LOAD_SYMBOL(lantern_abs_out)
  LOAD_SYMBOL(lantern_acos)
  LOAD_SYMBOL(lantern_acos_)
  LOAD_SYMBOL(lantern_acos_out)
  LOAD_SYMBOL(lantern_avg_pool1d)
  LOAD_SYMBOL(lantern_adaptive_avg_pool1d)
  LOAD_SYMBOL(lantern_adaptive_max_pool1d)
  LOAD_SYMBOL(lantern_add)
  LOAD_SYMBOL(lantern_add_)
  LOAD_SYMBOL(lantern_add_out)
  LOAD_SYMBOL(lantern_add)
  LOAD_SYMBOL(lantern_add_)
  LOAD_SYMBOL(lantern_addmv)
  LOAD_SYMBOL(lantern_addmv_)
  LOAD_SYMBOL(lantern_addmv_out)
  LOAD_SYMBOL(lantern_addr)
  LOAD_SYMBOL(lantern_addr_)
  LOAD_SYMBOL(lantern_addr_out)
  LOAD_SYMBOL(lantern_affine_grid_generator)
  LOAD_SYMBOL(lantern_affine_grid_generator_backward)
  LOAD_SYMBOL(lantern_all)
  LOAD_SYMBOL(lantern_all_out)
  LOAD_SYMBOL(lantern_all)
  LOAD_SYMBOL(lantern_all_out)
  LOAD_SYMBOL(lantern_allclose)
  LOAD_SYMBOL(lantern_any)
  LOAD_SYMBOL(lantern_any_out)
  LOAD_SYMBOL(lantern_any)
  LOAD_SYMBOL(lantern_any_out)
  LOAD_SYMBOL(lantern_arange)
  LOAD_SYMBOL(lantern_arange)
  LOAD_SYMBOL(lantern_arange)
  LOAD_SYMBOL(lantern_arange_out)
  LOAD_SYMBOL(lantern_arange_out)
  LOAD_SYMBOL(lantern__dim_arange)
  LOAD_SYMBOL(lantern_argmax)
  LOAD_SYMBOL(lantern_argmin)
  LOAD_SYMBOL(lantern_as_strided)
  LOAD_SYMBOL(lantern_as_strided_)
  LOAD_SYMBOL(lantern_asin)
  LOAD_SYMBOL(lantern_asin_)
  LOAD_SYMBOL(lantern_asin_out)
  LOAD_SYMBOL(lantern_atan)
  LOAD_SYMBOL(lantern_atan_)
  LOAD_SYMBOL(lantern_atan_out)
  LOAD_SYMBOL(lantern_baddbmm)
  LOAD_SYMBOL(lantern_baddbmm_)
  LOAD_SYMBOL(lantern__baddbmm_mkl_)
  LOAD_SYMBOL(lantern_baddbmm_out)
  LOAD_SYMBOL(lantern_bartlett_window)
  LOAD_SYMBOL(lantern_bartlett_window)
  LOAD_SYMBOL(lantern_batch_norm)
  LOAD_SYMBOL(lantern__batch_norm_impl_index)
  LOAD_SYMBOL(lantern__batch_norm_impl_index_backward)
  LOAD_SYMBOL(lantern_bernoulli)
  LOAD_SYMBOL(lantern_bernoulli_out)
  LOAD_SYMBOL(lantern_bernoulli_)
  LOAD_SYMBOL(lantern_bernoulli_)
  LOAD_SYMBOL(lantern_bernoulli)
  LOAD_SYMBOL(lantern_bilinear)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits)
  LOAD_SYMBOL(lantern_binary_cross_entropy_with_logits_backward)
  LOAD_SYMBOL(lantern_bincount)
  LOAD_SYMBOL(lantern_bitwise_not)
  LOAD_SYMBOL(lantern_bitwise_not_)
  LOAD_SYMBOL(lantern_bitwise_not_out)
  LOAD_SYMBOL(lantern_logical_not)
  LOAD_SYMBOL(lantern_logical_not_)
  LOAD_SYMBOL(lantern_logical_not_out)
  LOAD_SYMBOL(lantern_logical_xor)
  LOAD_SYMBOL(lantern_logical_xor_)
  LOAD_SYMBOL(lantern_logical_xor_out)
  LOAD_SYMBOL(lantern_blackman_window)
  LOAD_SYMBOL(lantern_blackman_window)
  LOAD_SYMBOL(lantern_bmm)
  LOAD_SYMBOL(lantern_bmm_out)
  LOAD_SYMBOL(lantern_broadcast_tensors)
  LOAD_SYMBOL(lantern_cat)
  LOAD_SYMBOL(lantern_cat_out)
  LOAD_SYMBOL(lantern_cat)
  LOAD_SYMBOL(lantern_cat_out)
  LOAD_SYMBOL(lantern_ceil)
  LOAD_SYMBOL(lantern_ceil_)
  LOAD_SYMBOL(lantern_ceil_out)
  LOAD_SYMBOL(lantern_chain_matmul)
  LOAD_SYMBOL(lantern_chunk)
  LOAD_SYMBOL(lantern_clamp)
  LOAD_SYMBOL(lantern_clamp_)
  LOAD_SYMBOL(lantern_clamp_out)
  LOAD_SYMBOL(lantern_clamp_max)
  LOAD_SYMBOL(lantern_clamp_max_)
  LOAD_SYMBOL(lantern_clamp_max_out)
  LOAD_SYMBOL(lantern_clamp_min)
  LOAD_SYMBOL(lantern_clamp_min_)
  LOAD_SYMBOL(lantern_clamp_min_out)
  LOAD_SYMBOL(lantern_cudnn_is_acceptable)
  LOAD_SYMBOL(lantern_constant_pad_nd)
  LOAD_SYMBOL(lantern_contiguous)
  LOAD_SYMBOL(lantern_convolution)
  LOAD_SYMBOL(lantern_convolution_overrideable)
  LOAD_SYMBOL(lantern_convolution_backward_overrideable)
  LOAD_SYMBOL(lantern__convolution)
  LOAD_SYMBOL(lantern__convolution_nogroup)
  LOAD_SYMBOL(lantern__convolution_double_backward)
  LOAD_SYMBOL(lantern_conv1d)
  LOAD_SYMBOL(lantern_conv2d)
  LOAD_SYMBOL(lantern_conv3d)
  LOAD_SYMBOL(lantern_conv_tbc)
  LOAD_SYMBOL(lantern_conv_tbc_backward)
  LOAD_SYMBOL(lantern_conv_transpose1d)
  LOAD_SYMBOL(lantern_conv_transpose2d)
  LOAD_SYMBOL(lantern_conv_transpose3d)
  LOAD_SYMBOL(lantern_copy_)
  LOAD_SYMBOL(lantern__copy_from)
  LOAD_SYMBOL(lantern_cos)
  LOAD_SYMBOL(lantern_cos_)
  LOAD_SYMBOL(lantern_cos_out)
  LOAD_SYMBOL(lantern_cosh)
  LOAD_SYMBOL(lantern_cosh_)
  LOAD_SYMBOL(lantern_cosh_out)
  LOAD_SYMBOL(lantern_cosine_embedding_loss)
  LOAD_SYMBOL(lantern_cudnn_affine_grid_generator)
  LOAD_SYMBOL(lantern_cudnn_affine_grid_generator_backward)
  LOAD_SYMBOL(lantern_cudnn_batch_norm)
  LOAD_SYMBOL(lantern_cudnn_batch_norm_backward)
  LOAD_SYMBOL(lantern_cudnn_convolution)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_input)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_bias)
  LOAD_SYMBOL(lantern_cudnn_convolution_backward_weight)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_bias)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_input)
  LOAD_SYMBOL(lantern_cudnn_convolution_transpose_backward_weight)
  LOAD_SYMBOL(lantern_cudnn_grid_sampler)
  LOAD_SYMBOL(lantern_cudnn_grid_sampler_backward)
  LOAD_SYMBOL(lantern_cumsum)
  LOAD_SYMBOL(lantern_cumsum_out)
  LOAD_SYMBOL(lantern_cumsum)
  LOAD_SYMBOL(lantern_cumsum_out)
  LOAD_SYMBOL(lantern_cumprod)
  LOAD_SYMBOL(lantern_cumprod_out)
  LOAD_SYMBOL(lantern_cumprod)
  LOAD_SYMBOL(lantern_cumprod_out)
  LOAD_SYMBOL(lantern_ctc_loss)
  LOAD_SYMBOL(lantern_ctc_loss)
  LOAD_SYMBOL(lantern__ctc_loss)
  LOAD_SYMBOL(lantern__ctc_loss_backward)
  LOAD_SYMBOL(lantern_det)
  LOAD_SYMBOL(lantern_diag_embed)
  LOAD_SYMBOL(lantern_diagflat)
  LOAD_SYMBOL(lantern_diagonal)
  LOAD_SYMBOL(lantern_fill_diagonal_)
  LOAD_SYMBOL(lantern_div)
  LOAD_SYMBOL(lantern_div_)
  LOAD_SYMBOL(lantern_div_out)
  LOAD_SYMBOL(lantern_div)
  LOAD_SYMBOL(lantern_div_)
  LOAD_SYMBOL(lantern_dot)
  LOAD_SYMBOL(lantern_dot_out)
  LOAD_SYMBOL(lantern_einsum)
  LOAD_SYMBOL(lantern_embedding)
  LOAD_SYMBOL(lantern_embedding_backward)
  LOAD_SYMBOL(lantern_embedding_dense_backward)
  LOAD_SYMBOL(lantern_embedding_renorm_)
  LOAD_SYMBOL(lantern_embedding_sparse_backward)
  LOAD_SYMBOL(lantern_embedding_bag)
  LOAD_SYMBOL(lantern__embedding_bag)
  LOAD_SYMBOL(lantern__embedding_bag_backward)
  LOAD_SYMBOL(lantern__embedding_bag_sparse_backward)
  LOAD_SYMBOL(lantern__embedding_bag_dense_backward)
  LOAD_SYMBOL(lantern__embedding_bag_per_sample_weights_backward)
  LOAD_SYMBOL(lantern_empty)
  LOAD_SYMBOL(lantern_empty)
  LOAD_SYMBOL(lantern_new_empty)
  LOAD_SYMBOL(lantern_new_full)
  LOAD_SYMBOL(lantern__empty_affine_quantized)
  LOAD_SYMBOL(lantern__empty_per_channel_affine_quantized)
  LOAD_SYMBOL(lantern_resize_)
  LOAD_SYMBOL(lantern_empty_out)
  LOAD_SYMBOL(lantern_empty_like)
  LOAD_SYMBOL(lantern_empty_like)
  LOAD_SYMBOL(lantern_empty_strided)
  LOAD_SYMBOL(lantern_erf)
  LOAD_SYMBOL(lantern_erf_)
  LOAD_SYMBOL(lantern_erf_out)
  LOAD_SYMBOL(lantern_erfc)
  LOAD_SYMBOL(lantern_erfc_)
  LOAD_SYMBOL(lantern_erfc_out)
  LOAD_SYMBOL(lantern_exp)
  LOAD_SYMBOL(lantern_exp_)
  LOAD_SYMBOL(lantern_exp_out)
  LOAD_SYMBOL(lantern_expm1)
  LOAD_SYMBOL(lantern_expm1_)
  LOAD_SYMBOL(lantern_expm1_out)
  LOAD_SYMBOL(lantern_expand)
  LOAD_SYMBOL(lantern_expand_as)
  LOAD_SYMBOL(lantern_eye)
  LOAD_SYMBOL(lantern_eye)
  LOAD_SYMBOL(lantern_eye_out)
  LOAD_SYMBOL(lantern_eye_out)
  LOAD_SYMBOL(lantern_flatten)
  LOAD_SYMBOL(lantern_flatten)
  LOAD_SYMBOL(lantern_flatten)
  LOAD_SYMBOL(lantern_flatten)
  LOAD_SYMBOL(lantern_fill_)
  LOAD_SYMBOL(lantern_fill_)
  LOAD_SYMBOL(lantern_floor)
  LOAD_SYMBOL(lantern_floor_)
  LOAD_SYMBOL(lantern_floor_out)
  LOAD_SYMBOL(lantern_frac)
  LOAD_SYMBOL(lantern_frac_)
  LOAD_SYMBOL(lantern_frac_out)
  LOAD_SYMBOL(lantern_full)
  LOAD_SYMBOL(lantern_full)
  LOAD_SYMBOL(lantern_full_out)
  LOAD_SYMBOL(lantern_full_like)
  LOAD_SYMBOL(lantern_full_like)
  LOAD_SYMBOL(lantern_from_file)
  LOAD_SYMBOL(lantern_grid_sampler)
  LOAD_SYMBOL(lantern_grid_sampler_2d)
  LOAD_SYMBOL(lantern_grid_sampler_2d_backward)
  LOAD_SYMBOL(lantern_grid_sampler_3d)
  LOAD_SYMBOL(lantern_grid_sampler_3d_backward)
  LOAD_SYMBOL(lantern_hann_window)
  LOAD_SYMBOL(lantern_hann_window)
  LOAD_SYMBOL(lantern_hamming_window)
  LOAD_SYMBOL(lantern_hamming_window)
  LOAD_SYMBOL(lantern_hamming_window)
  LOAD_SYMBOL(lantern_hamming_window)
  LOAD_SYMBOL(lantern_hinge_embedding_loss)
  LOAD_SYMBOL(lantern_ger)
  LOAD_SYMBOL(lantern_ger_out)
  LOAD_SYMBOL(lantern_group_norm)
  LOAD_SYMBOL(lantern_fft)
  LOAD_SYMBOL(lantern_ifft)
  LOAD_SYMBOL(lantern_rfft)
  LOAD_SYMBOL(lantern_irfft)
  LOAD_SYMBOL(lantern__fft_with_size)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_size)
  LOAD_SYMBOL(lantern__cufft_get_plan_cache_max_size)
  LOAD_SYMBOL(lantern__cufft_set_plan_cache_max_size)
  LOAD_SYMBOL(lantern__cufft_clear_plan_cache)
  LOAD_SYMBOL(lantern_index)
  LOAD_SYMBOL(lantern_index_copy_)
  LOAD_SYMBOL(lantern_index_copy)
  LOAD_SYMBOL(lantern_index_copy_)
  LOAD_SYMBOL(lantern_index_copy)
  LOAD_SYMBOL(lantern_index_put_)
  LOAD_SYMBOL(lantern_index_put)
  LOAD_SYMBOL(lantern__index_put_impl_)
  LOAD_SYMBOL(lantern_instance_norm)
  LOAD_SYMBOL(lantern_inverse)
  LOAD_SYMBOL(lantern_inverse_out)
  LOAD_SYMBOL(lantern__inverse_helper)
  LOAD_SYMBOL(lantern_isclose)
  LOAD_SYMBOL(lantern_isnan)
  LOAD_SYMBOL(lantern_is_distributed)
  LOAD_SYMBOL(lantern_is_floating_point)
  LOAD_SYMBOL(lantern_is_complex)
  LOAD_SYMBOL(lantern_is_nonzero)
  LOAD_SYMBOL(lantern_is_same_size)
  LOAD_SYMBOL(lantern_is_signed)
  LOAD_SYMBOL(lantern_kl_div)
  LOAD_SYMBOL(lantern_kl_div_backward)
  LOAD_SYMBOL(lantern_kthvalue)
  LOAD_SYMBOL(lantern_kthvalue_out)
  LOAD_SYMBOL(lantern_kthvalue)
  LOAD_SYMBOL(lantern_kthvalue_out)
  LOAD_SYMBOL(lantern_layer_norm)
  LOAD_SYMBOL(lantern_native_layer_norm)
  LOAD_SYMBOL(lantern_native_layer_norm_backward)
  LOAD_SYMBOL(lantern_native_layer_norm_double_backward)
  LOAD_SYMBOL(lantern_linear)
  LOAD_SYMBOL(lantern_mkldnn_linear)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight_fp32_activation)
  LOAD_SYMBOL(lantern_fbgemm_linear_int8_weight)
  LOAD_SYMBOL(lantern_fbgemm_linear_quantize_weight)
  LOAD_SYMBOL(lantern_fbgemm_pack_gemm_matrix_fp16)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight_fp32_activation)
  LOAD_SYMBOL(lantern_fbgemm_linear_fp16_weight)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix)
  LOAD_SYMBOL(lantern_fbgemm_pack_quantized_matrix)
  LOAD_SYMBOL(lantern_linspace)
  LOAD_SYMBOL(lantern_linspace_out)
  LOAD_SYMBOL(lantern_log)
  LOAD_SYMBOL(lantern_log_)
  LOAD_SYMBOL(lantern_log_out)
  LOAD_SYMBOL(lantern_log10)
  LOAD_SYMBOL(lantern_log10_)
  LOAD_SYMBOL(lantern_log10_out)
  LOAD_SYMBOL(lantern_log1p)
  LOAD_SYMBOL(lantern_log1p_)
  LOAD_SYMBOL(lantern_log1p_out)
  LOAD_SYMBOL(lantern_log2)
  LOAD_SYMBOL(lantern_log2_)
  LOAD_SYMBOL(lantern_log2_out)
  LOAD_SYMBOL(lantern_logdet)
  LOAD_SYMBOL(lantern_logspace)
  LOAD_SYMBOL(lantern_logspace_out)
  LOAD_SYMBOL(lantern_log_softmax)
  LOAD_SYMBOL(lantern_log_softmax)
  LOAD_SYMBOL(lantern__log_softmax)
  LOAD_SYMBOL(lantern__log_softmax_backward_data)
  LOAD_SYMBOL(lantern_logsumexp)
  LOAD_SYMBOL(lantern_logsumexp_out)
  LOAD_SYMBOL(lantern_logsumexp)
  LOAD_SYMBOL(lantern_logsumexp_out)
  LOAD_SYMBOL(lantern_margin_ranking_loss)
  LOAD_SYMBOL(lantern_matmul)
  LOAD_SYMBOL(lantern_matmul_out)
  LOAD_SYMBOL(lantern_matrix_rank)
  LOAD_SYMBOL(lantern_matrix_rank)
  LOAD_SYMBOL(lantern_matrix_power)
  LOAD_SYMBOL(lantern_max)
  LOAD_SYMBOL(lantern_max_out)
  LOAD_SYMBOL(lantern_max_values)
  LOAD_SYMBOL(lantern_max)
  LOAD_SYMBOL(lantern_max_out)
  LOAD_SYMBOL(lantern_max_values)
  LOAD_SYMBOL(lantern_max_pool1d_with_indices)
  LOAD_SYMBOL(lantern_max_pool1d)
  LOAD_SYMBOL(lantern_max_pool2d)
  LOAD_SYMBOL(lantern_mkldnn_max_pool2d)
  LOAD_SYMBOL(lantern_quantized_max_pool2d)
  LOAD_SYMBOL(lantern_max_pool3d)
  LOAD_SYMBOL(lantern_mean)
  LOAD_SYMBOL(lantern_mean)
  LOAD_SYMBOL(lantern_mean_out)
  LOAD_SYMBOL(lantern_mean)
  LOAD_SYMBOL(lantern_mean_out)
  LOAD_SYMBOL(lantern_median)
  LOAD_SYMBOL(lantern_median_out)
  LOAD_SYMBOL(lantern_median)
  LOAD_SYMBOL(lantern_median_out)
  LOAD_SYMBOL(lantern_min)
  LOAD_SYMBOL(lantern_min_out)
  LOAD_SYMBOL(lantern_min_values)
  LOAD_SYMBOL(lantern_min)
  LOAD_SYMBOL(lantern_min_out)
  LOAD_SYMBOL(lantern_min_values)
  LOAD_SYMBOL(lantern_mkldnn_convolution)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_input)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward_weights)
  LOAD_SYMBOL(lantern_mkldnn_convolution_backward)
  LOAD_SYMBOL(lantern_miopen_batch_norm)
  LOAD_SYMBOL(lantern_miopen_batch_norm_backward)
  LOAD_SYMBOL(lantern_miopen_convolution)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_input)
  LOAD_SYMBOL(lantern_miopen_convolution_backward)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_bias)
  LOAD_SYMBOL(lantern_miopen_convolution_backward_weight)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward_input)
  LOAD_SYMBOL(lantern_miopen_convolution_transpose_backward_weight)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward_input)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward)
  LOAD_SYMBOL(lantern_miopen_depthwise_convolution_backward_weight)
  LOAD_SYMBOL(lantern_miopen_rnn)
  LOAD_SYMBOL(lantern_miopen_rnn_backward)
  LOAD_SYMBOL(lantern_mm)
  LOAD_SYMBOL(lantern_mm_out)
  LOAD_SYMBOL(lantern__sparse_mm)
  LOAD_SYMBOL(lantern_mode)
  LOAD_SYMBOL(lantern_mode_out)
  LOAD_SYMBOL(lantern_mode)
  LOAD_SYMBOL(lantern_mode_out)
  LOAD_SYMBOL(lantern_mul)
  LOAD_SYMBOL(lantern_mul_)
  LOAD_SYMBOL(lantern_mul_out)
  LOAD_SYMBOL(lantern_mul)
  LOAD_SYMBOL(lantern_mul_)
  LOAD_SYMBOL(lantern_mv)
  LOAD_SYMBOL(lantern_mv_out)
  LOAD_SYMBOL(lantern_mvlgamma)
  LOAD_SYMBOL(lantern_mvlgamma_)
  LOAD_SYMBOL(lantern_narrow_copy)
  LOAD_SYMBOL(lantern_narrow)
  LOAD_SYMBOL(lantern_native_batch_norm)
  LOAD_SYMBOL(lantern_batch_norm_stats)
  LOAD_SYMBOL(lantern_batch_norm_elemt)
  LOAD_SYMBOL(lantern_batch_norm_gather_stats)
  LOAD_SYMBOL(lantern_batch_norm_gather_stats_with_counts)
  LOAD_SYMBOL(lantern_native_batch_norm_backward)
  LOAD_SYMBOL(lantern_batch_norm_backward_reduce)
  LOAD_SYMBOL(lantern_batch_norm_backward_elemt)
  LOAD_SYMBOL(lantern_batch_norm_update_stats)
  LOAD_SYMBOL(lantern__nnpack_available)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward_input)
  LOAD_SYMBOL(lantern__nnpack_spatial_convolution_backward_weight)
  LOAD_SYMBOL(lantern_ones)
  LOAD_SYMBOL(lantern_ones)
  LOAD_SYMBOL(lantern_ones_out)
  LOAD_SYMBOL(lantern_ones_like)
  LOAD_SYMBOL(lantern_ones_like)
  LOAD_SYMBOL(lantern_pairwise_distance)
  LOAD_SYMBOL(lantern_cdist)
  LOAD_SYMBOL(lantern__cdist_backward)
  LOAD_SYMBOL(lantern_pdist)
  LOAD_SYMBOL(lantern__pdist_forward)
  LOAD_SYMBOL(lantern__pdist_backward)
  LOAD_SYMBOL(lantern_cosine_similarity)
  LOAD_SYMBOL(lantern_permute)
  LOAD_SYMBOL(lantern_numpy_t)
  LOAD_SYMBOL(lantern_pixel_shuffle)
  LOAD_SYMBOL(lantern_is_pinned)
  LOAD_SYMBOL(lantern_pin_memory)
  LOAD_SYMBOL(lantern_pinverse)
  LOAD_SYMBOL(lantern_poisson_nll_loss)
  LOAD_SYMBOL(lantern_scalar_tensor)
  LOAD_SYMBOL(lantern_rand)
  LOAD_SYMBOL(lantern_rand)
  LOAD_SYMBOL(lantern_rand)
  LOAD_SYMBOL(lantern_rand)
  LOAD_SYMBOL(lantern_rand_out)
  LOAD_SYMBOL(lantern_rand_out)
  LOAD_SYMBOL(lantern_rand_like)
  LOAD_SYMBOL(lantern_rand_like)
  LOAD_SYMBOL(lantern_randint)
  LOAD_SYMBOL(lantern_randint)
  LOAD_SYMBOL(lantern_randint)
  LOAD_SYMBOL(lantern_randint)
  LOAD_SYMBOL(lantern_randint_out)
  LOAD_SYMBOL(lantern_randint_out)
  LOAD_SYMBOL(lantern_randint_out)
  LOAD_SYMBOL(lantern_randint_out)
  LOAD_SYMBOL(lantern_randint_like)
  LOAD_SYMBOL(lantern_randint_like)
  LOAD_SYMBOL(lantern_randint_like)
  LOAD_SYMBOL(lantern_randint_like)
  LOAD_SYMBOL(lantern_randn)
  LOAD_SYMBOL(lantern_randn)
  LOAD_SYMBOL(lantern_randn)
  LOAD_SYMBOL(lantern_randn)
  LOAD_SYMBOL(lantern_randn_out)
  LOAD_SYMBOL(lantern_randn_out)
  LOAD_SYMBOL(lantern_randn_like)
  LOAD_SYMBOL(lantern_randn_like)
  LOAD_SYMBOL(lantern_randperm)
  LOAD_SYMBOL(lantern_randperm)
  LOAD_SYMBOL(lantern_randperm_out)
  LOAD_SYMBOL(lantern_randperm_out)
  LOAD_SYMBOL(lantern_range)
  LOAD_SYMBOL(lantern_range)
  LOAD_SYMBOL(lantern_range_out)
  LOAD_SYMBOL(lantern_reciprocal)
  LOAD_SYMBOL(lantern_reciprocal_)
  LOAD_SYMBOL(lantern_reciprocal_out)
  LOAD_SYMBOL(lantern_neg)
  LOAD_SYMBOL(lantern_neg_)
  LOAD_SYMBOL(lantern_neg_out)
  LOAD_SYMBOL(lantern_repeat)
  LOAD_SYMBOL(lantern_repeat_interleave)
  LOAD_SYMBOL(lantern_repeat_interleave)
  LOAD_SYMBOL(lantern_repeat_interleave)
  LOAD_SYMBOL(lantern_reshape)
  LOAD_SYMBOL(lantern__mkldnn_reshape)
  LOAD_SYMBOL(lantern_reshape_as)
  LOAD_SYMBOL(lantern_round)
  LOAD_SYMBOL(lantern_round_)
  LOAD_SYMBOL(lantern_round_out)
  LOAD_SYMBOL(lantern_rrelu)
  LOAD_SYMBOL(lantern_rrelu_)
  LOAD_SYMBOL(lantern_relu)
  LOAD_SYMBOL(lantern_relu_)
  LOAD_SYMBOL(lantern_prelu)
  LOAD_SYMBOL(lantern_prelu_backward)
  LOAD_SYMBOL(lantern_gelu)
  LOAD_SYMBOL(lantern_gelu_backward)
  LOAD_SYMBOL(lantern_hardshrink)
  LOAD_SYMBOL(lantern_hardshrink_backward)
  LOAD_SYMBOL(lantern_rsqrt)
  LOAD_SYMBOL(lantern_rsqrt_)
  LOAD_SYMBOL(lantern_rsqrt_out)
  LOAD_SYMBOL(lantern_select)
  LOAD_SYMBOL(lantern_select)
  LOAD_SYMBOL(lantern_selu)
  LOAD_SYMBOL(lantern_selu_)
  LOAD_SYMBOL(lantern_celu)
  LOAD_SYMBOL(lantern_celu_)
  LOAD_SYMBOL(lantern_sigmoid)
  LOAD_SYMBOL(lantern_sigmoid_)
  LOAD_SYMBOL(lantern_sigmoid_out)
  LOAD_SYMBOL(lantern_sin)
  LOAD_SYMBOL(lantern_sin_)
  LOAD_SYMBOL(lantern_sin_out)
  LOAD_SYMBOL(lantern_sinh)
  LOAD_SYMBOL(lantern_sinh_)
  LOAD_SYMBOL(lantern_sinh_out)
  LOAD_SYMBOL(lantern_detach)
  LOAD_SYMBOL(lantern_detach_)
  LOAD_SYMBOL(lantern_size)
  LOAD_SYMBOL(lantern_size)
  LOAD_SYMBOL(lantern_slice)
  LOAD_SYMBOL(lantern_slogdet)
  LOAD_SYMBOL(lantern_smm)
  LOAD_SYMBOL(lantern_softmax)
  LOAD_SYMBOL(lantern_softmax)
  LOAD_SYMBOL(lantern__softmax)
  LOAD_SYMBOL(lantern__softmax_backward_data)
  LOAD_SYMBOL(lantern_split)
  LOAD_SYMBOL(lantern_split_with_sizes)
  LOAD_SYMBOL(lantern_squeeze)
  LOAD_SYMBOL(lantern_squeeze)
  LOAD_SYMBOL(lantern_squeeze)
  LOAD_SYMBOL(lantern_squeeze_)
  LOAD_SYMBOL(lantern_squeeze_)
  LOAD_SYMBOL(lantern_squeeze_)
  LOAD_SYMBOL(lantern_sspaddmm)
  LOAD_SYMBOL(lantern_sspaddmm_out)
  LOAD_SYMBOL(lantern_stack)
  LOAD_SYMBOL(lantern_stack_out)
  LOAD_SYMBOL(lantern_stft)
  LOAD_SYMBOL(lantern_stride)
  LOAD_SYMBOL(lantern_stride)
  LOAD_SYMBOL(lantern_sum)
  LOAD_SYMBOL(lantern_sum)
  LOAD_SYMBOL(lantern_sum)
  LOAD_SYMBOL(lantern_sum_out)
  LOAD_SYMBOL(lantern_sum_out)
  LOAD_SYMBOL(lantern_sum_to_size)
  LOAD_SYMBOL(lantern_sqrt)
  LOAD_SYMBOL(lantern_sqrt_)
  LOAD_SYMBOL(lantern_sqrt_out)
  LOAD_SYMBOL(lantern_std)
  LOAD_SYMBOL(lantern_std)
  LOAD_SYMBOL(lantern_std_mean)
  LOAD_SYMBOL(lantern_std_mean)
  LOAD_SYMBOL(lantern_std_mean)
  LOAD_SYMBOL(lantern_std_out)
  LOAD_SYMBOL(lantern_std)
  LOAD_SYMBOL(lantern_std_out)
  LOAD_SYMBOL(lantern_prod)
  LOAD_SYMBOL(lantern_prod)
  LOAD_SYMBOL(lantern_prod_out)
  LOAD_SYMBOL(lantern_prod)
  LOAD_SYMBOL(lantern_prod_out)
  LOAD_SYMBOL(lantern_t)
  LOAD_SYMBOL(lantern_t_)
  LOAD_SYMBOL(lantern_tan)
  LOAD_SYMBOL(lantern_tan_)
  LOAD_SYMBOL(lantern_tan_out)
  LOAD_SYMBOL(lantern_tanh)
  LOAD_SYMBOL(lantern_tanh_)
  LOAD_SYMBOL(lantern_tanh_out)
  LOAD_SYMBOL(lantern_tensordot)
  LOAD_SYMBOL(lantern_threshold)
  LOAD_SYMBOL(lantern_threshold_)
  LOAD_SYMBOL(lantern_threshold_out)
  LOAD_SYMBOL(lantern_threshold_backward)
  LOAD_SYMBOL(lantern_transpose)
  LOAD_SYMBOL(lantern_transpose)
  LOAD_SYMBOL(lantern__mkldnn_transpose)
  LOAD_SYMBOL(lantern_transpose_)
  LOAD_SYMBOL(lantern__mkldnn_transpose_)
  LOAD_SYMBOL(lantern_one_hot)
  LOAD_SYMBOL(lantern_flip)
  LOAD_SYMBOL(lantern_roll)
  LOAD_SYMBOL(lantern_rot90)
  LOAD_SYMBOL(lantern_trapz)
  LOAD_SYMBOL(lantern_trapz)
  LOAD_SYMBOL(lantern__trilinear)
  LOAD_SYMBOL(lantern_triplet_margin_loss)
  LOAD_SYMBOL(lantern_trunc)
  LOAD_SYMBOL(lantern_trunc_)
  LOAD_SYMBOL(lantern_trunc_out)
  LOAD_SYMBOL(lantern_type_as)
  LOAD_SYMBOL(lantern__has_compatible_shallow_copy_type)
  LOAD_SYMBOL(lantern__unique)
  LOAD_SYMBOL(lantern_unique_dim)
  LOAD_SYMBOL(lantern_unique_consecutive)
  LOAD_SYMBOL(lantern_unique_dim_consecutive)
  LOAD_SYMBOL(lantern__unique2)
  LOAD_SYMBOL(lantern__unsafe_view)
  LOAD_SYMBOL(lantern_unsqueeze)
  LOAD_SYMBOL(lantern_unsqueeze_)
  LOAD_SYMBOL(lantern_var)
  LOAD_SYMBOL(lantern_var)
  LOAD_SYMBOL(lantern_var_out)
  LOAD_SYMBOL(lantern_var)
  LOAD_SYMBOL(lantern_var_out)
  LOAD_SYMBOL(lantern_var_mean)
  LOAD_SYMBOL(lantern_var_mean)
  LOAD_SYMBOL(lantern_var_mean)
  LOAD_SYMBOL(lantern_view_as)
  LOAD_SYMBOL(lantern_where)
  LOAD_SYMBOL(lantern_where)
  LOAD_SYMBOL(lantern__s_where)
  LOAD_SYMBOL(lantern_norm_except_dim)
  LOAD_SYMBOL(lantern__weight_norm)
  LOAD_SYMBOL(lantern__weight_norm_cuda_interface)
  LOAD_SYMBOL(lantern__weight_norm_cuda_interface_backward)
  LOAD_SYMBOL(lantern__weight_norm_differentiable_backward)
  LOAD_SYMBOL(lantern_zeros)
  LOAD_SYMBOL(lantern_zeros)
  LOAD_SYMBOL(lantern_zeros_out)
  LOAD_SYMBOL(lantern_zeros_like)
  LOAD_SYMBOL(lantern_zeros_like)
  LOAD_SYMBOL(lantern__standard_gamma_grad)
  LOAD_SYMBOL(lantern__standard_gamma)
  LOAD_SYMBOL(lantern__dirichlet_grad)
  LOAD_SYMBOL(lantern__sample_dirichlet)
  LOAD_SYMBOL(lantern_poisson)
  LOAD_SYMBOL(lantern_native_norm)
  LOAD_SYMBOL(lantern__sparse_sum)
  LOAD_SYMBOL(lantern__sparse_sum)
  LOAD_SYMBOL(lantern__sparse_sum)
  LOAD_SYMBOL(lantern__sparse_sum)
  LOAD_SYMBOL(lantern__sparse_sum_backward)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm_out)
  LOAD_SYMBOL(lantern_norm_out)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm)
  LOAD_SYMBOL(lantern_norm_out)
  LOAD_SYMBOL(lantern_norm_out)
  LOAD_SYMBOL(lantern_frobenius_norm)
  LOAD_SYMBOL(lantern_frobenius_norm)
  LOAD_SYMBOL(lantern_frobenius_norm_out)
  LOAD_SYMBOL(lantern_nuclear_norm)
  LOAD_SYMBOL(lantern_nuclear_norm_out)
  LOAD_SYMBOL(lantern_nuclear_norm)
  LOAD_SYMBOL(lantern_nuclear_norm_out)
  LOAD_SYMBOL(lantern_clone)
  LOAD_SYMBOL(lantern_resize_as_)
  LOAD_SYMBOL(lantern_pow_out)
  LOAD_SYMBOL(lantern_pow)
  LOAD_SYMBOL(lantern_zero_)
  LOAD_SYMBOL(lantern_sub_out)
  LOAD_SYMBOL(lantern_sub)
  LOAD_SYMBOL(lantern_sub_)
  LOAD_SYMBOL(lantern_sub)
  LOAD_SYMBOL(lantern_sub_)
  LOAD_SYMBOL(lantern_rsub)
  LOAD_SYMBOL(lantern_rsub)
  LOAD_SYMBOL(lantern__sparse_addmm)
  LOAD_SYMBOL(lantern_addmm_out)
  LOAD_SYMBOL(lantern_addmm)
  LOAD_SYMBOL(lantern_addmm_)
  LOAD_SYMBOL(lantern_sparse_coo_tensor)
  LOAD_SYMBOL(lantern_sparse_coo_tensor)
  LOAD_SYMBOL(lantern_sparse_coo_tensor)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_unsafe)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_with_dims)
  LOAD_SYMBOL(lantern__sparse_coo_tensor_with_dims_and_tensors)
  LOAD_SYMBOL(lantern_sparse_resize_)
  LOAD_SYMBOL(lantern_sparse_resize_and_clear_)
  LOAD_SYMBOL(lantern_sparse_mask)
  LOAD_SYMBOL(lantern_to_dense)
  LOAD_SYMBOL(lantern_to_dense_backward)
  LOAD_SYMBOL(lantern_sparse_dim)
  LOAD_SYMBOL(lantern__dimi)
  LOAD_SYMBOL(lantern_dense_dim)
  LOAD_SYMBOL(lantern__dimv)
  LOAD_SYMBOL(lantern__nnz)
  LOAD_SYMBOL(lantern_coalesce)
  LOAD_SYMBOL(lantern_is_coalesced)
  LOAD_SYMBOL(lantern__indices)
  LOAD_SYMBOL(lantern__values)
  LOAD_SYMBOL(lantern__coalesced_)
  LOAD_SYMBOL(lantern_indices)
  LOAD_SYMBOL(lantern_values)
  LOAD_SYMBOL(lantern_hspmm_out)
  LOAD_SYMBOL(lantern_hspmm)
  LOAD_SYMBOL(lantern_copy_sparse_to_sparse_)
  LOAD_SYMBOL(lantern_numel)
  LOAD_SYMBOL(lantern_unbind)
  LOAD_SYMBOL(lantern_unbind)
  LOAD_SYMBOL(lantern_to_sparse)
  LOAD_SYMBOL(lantern_to_sparse)
  LOAD_SYMBOL(lantern_to_mkldnn)
  LOAD_SYMBOL(lantern_mkldnn_reorder_conv2d_weight)
  LOAD_SYMBOL(lantern_to_mkldnn_backward)
  LOAD_SYMBOL(lantern_quantize_per_tensor)
  LOAD_SYMBOL(lantern_quantize_per_channel)
  LOAD_SYMBOL(lantern_dequantize)
  LOAD_SYMBOL(lantern_q_scale)
  LOAD_SYMBOL(lantern_q_zero_point)
  LOAD_SYMBOL(lantern_q_per_channel_scales)
  LOAD_SYMBOL(lantern_q_per_channel_zero_points)
  LOAD_SYMBOL(lantern_q_per_channel_axis)
  LOAD_SYMBOL(lantern_int_repr)
  LOAD_SYMBOL(lantern__make_per_tensor_quantized_tensor)
  LOAD_SYMBOL(lantern__make_per_channel_quantized_tensor)
  LOAD_SYMBOL(lantern_qscheme)
  LOAD_SYMBOL(lantern_fake_quantize_per_tensor_affine)
  LOAD_SYMBOL(lantern_fake_quantize_per_tensor_affine_backward)
  LOAD_SYMBOL(lantern_fake_quantize_per_channel_affine)
  LOAD_SYMBOL(lantern_fake_quantize_per_channel_affine_backward)
  LOAD_SYMBOL(lantern_to)
  LOAD_SYMBOL(lantern_to)
  LOAD_SYMBOL(lantern_to)
  LOAD_SYMBOL(lantern_to)
  LOAD_SYMBOL(lantern_meshgrid)
  LOAD_SYMBOL(lantern_cartesian_prod)
  LOAD_SYMBOL(lantern_combinations)
  LOAD_SYMBOL(lantern_item)
  LOAD_SYMBOL(lantern_result_type)
  LOAD_SYMBOL(lantern_result_type)
  LOAD_SYMBOL(lantern_result_type)
  LOAD_SYMBOL(lantern_result_type)
  LOAD_SYMBOL(lantern_can_cast)
  LOAD_SYMBOL(lantern_promote_types)
  LOAD_SYMBOL(lantern__local_scalar_dense)
  LOAD_SYMBOL(lantern__thnn_fused_lstm_cell)
  LOAD_SYMBOL(lantern__thnn_fused_lstm_cell_backward)
  LOAD_SYMBOL(lantern__thnn_differentiable_lstm_cell_backward)
  LOAD_SYMBOL(lantern__thnn_fused_gru_cell)
  LOAD_SYMBOL(lantern__thnn_fused_gru_cell_backward)
  LOAD_SYMBOL(lantern__thnn_differentiable_gru_cell_backward)
  LOAD_SYMBOL(lantern_lstm)
  LOAD_SYMBOL(lantern_lstm)
  LOAD_SYMBOL(lantern_gru)
  LOAD_SYMBOL(lantern_gru)
  LOAD_SYMBOL(lantern_rnn_tanh)
  LOAD_SYMBOL(lantern_rnn_tanh)
  LOAD_SYMBOL(lantern_rnn_relu)
  LOAD_SYMBOL(lantern_rnn_relu)
  LOAD_SYMBOL(lantern_lstm_cell)
  LOAD_SYMBOL(lantern_gru_cell)
  LOAD_SYMBOL(lantern_rnn_tanh_cell)
  LOAD_SYMBOL(lantern_rnn_relu_cell)
  LOAD_SYMBOL(lantern_quantized_lstm)
  LOAD_SYMBOL(lantern_quantized_gru)
  LOAD_SYMBOL(lantern_quantized_gru)
  LOAD_SYMBOL(lantern_quantized_lstm_cell)
  LOAD_SYMBOL(lantern_quantized_gru_cell)
  LOAD_SYMBOL(lantern_quantized_rnn_relu_cell)
  LOAD_SYMBOL(lantern_quantized_rnn_tanh_cell)
  LOAD_SYMBOL(lantern__pack_padded_sequence)
  LOAD_SYMBOL(lantern__pack_padded_sequence_backward)
  LOAD_SYMBOL(lantern__pad_packed_sequence)
  LOAD_SYMBOL(lantern_set_)
  LOAD_SYMBOL(lantern_set_)
  LOAD_SYMBOL(lantern_set_)
  LOAD_SYMBOL(lantern_set_)
  LOAD_SYMBOL(lantern_set_quantizer_)
  LOAD_SYMBOL(lantern_is_set_to)
  LOAD_SYMBOL(lantern_masked_fill_)
  LOAD_SYMBOL(lantern_masked_fill)
  LOAD_SYMBOL(lantern_masked_fill_)
  LOAD_SYMBOL(lantern_masked_fill)
  LOAD_SYMBOL(lantern_masked_scatter_)
  LOAD_SYMBOL(lantern_masked_scatter)
  LOAD_SYMBOL(lantern_view)
  LOAD_SYMBOL(lantern_put_)
  LOAD_SYMBOL(lantern_index_add_)
  LOAD_SYMBOL(lantern_index_add)
  LOAD_SYMBOL(lantern_index_add)
  LOAD_SYMBOL(lantern_index_fill_)
  LOAD_SYMBOL(lantern_index_fill)
  LOAD_SYMBOL(lantern_index_fill_)
  LOAD_SYMBOL(lantern_index_fill)
  LOAD_SYMBOL(lantern_index_fill_)
  LOAD_SYMBOL(lantern_index_fill_)
  LOAD_SYMBOL(lantern_index_fill)
  LOAD_SYMBOL(lantern_index_fill)
  LOAD_SYMBOL(lantern_scatter_)
  LOAD_SYMBOL(lantern_scatter)
  LOAD_SYMBOL(lantern_scatter_)
  LOAD_SYMBOL(lantern_scatter)
  LOAD_SYMBOL(lantern_scatter)
  LOAD_SYMBOL(lantern_scatter)
  LOAD_SYMBOL(lantern_scatter_add_)
  LOAD_SYMBOL(lantern_scatter_add)
  LOAD_SYMBOL(lantern_scatter_add)
  LOAD_SYMBOL(lantern_lt_)
  LOAD_SYMBOL(lantern_lt_)
  LOAD_SYMBOL(lantern_gt_)
  LOAD_SYMBOL(lantern_gt_)
  LOAD_SYMBOL(lantern_le_)
  LOAD_SYMBOL(lantern_le_)
  LOAD_SYMBOL(lantern_ge_)
  LOAD_SYMBOL(lantern_ge_)
  LOAD_SYMBOL(lantern_eq_)
  LOAD_SYMBOL(lantern_eq_)
  LOAD_SYMBOL(lantern_ne_)
  LOAD_SYMBOL(lantern_ne_)
  LOAD_SYMBOL(lantern___and__)
  LOAD_SYMBOL(lantern___and__)
  LOAD_SYMBOL(lantern___iand__)
  LOAD_SYMBOL(lantern___iand__)
  LOAD_SYMBOL(lantern___or__)
  LOAD_SYMBOL(lantern___or__)
  LOAD_SYMBOL(lantern___ior__)
  LOAD_SYMBOL(lantern___ior__)
  LOAD_SYMBOL(lantern___xor__)
  LOAD_SYMBOL(lantern___xor__)
  LOAD_SYMBOL(lantern___ixor__)
  LOAD_SYMBOL(lantern___ixor__)
  LOAD_SYMBOL(lantern___lshift__)
  LOAD_SYMBOL(lantern___lshift__)
  LOAD_SYMBOL(lantern___ilshift__)
  LOAD_SYMBOL(lantern___ilshift__)
  LOAD_SYMBOL(lantern___rshift__)
  LOAD_SYMBOL(lantern___rshift__)
  LOAD_SYMBOL(lantern___irshift__)
  LOAD_SYMBOL(lantern___irshift__)
  LOAD_SYMBOL(lantern_lgamma_)
  LOAD_SYMBOL(lantern_atan2_)
  LOAD_SYMBOL(lantern_tril_)
  LOAD_SYMBOL(lantern_triu_)
  LOAD_SYMBOL(lantern_digamma_)
  LOAD_SYMBOL(lantern_polygamma_)
  LOAD_SYMBOL(lantern_renorm_)
  LOAD_SYMBOL(lantern_pow_)
  LOAD_SYMBOL(lantern_pow_)
  LOAD_SYMBOL(lantern_lerp_)
  LOAD_SYMBOL(lantern_lerp_)
  LOAD_SYMBOL(lantern_fmod_)
  LOAD_SYMBOL(lantern_fmod_)
  LOAD_SYMBOL(lantern_remainder_)
  LOAD_SYMBOL(lantern_remainder_)
  LOAD_SYMBOL(lantern_addbmm_)
  LOAD_SYMBOL(lantern_addbmm_out)
  LOAD_SYMBOL(lantern_addbmm)
  LOAD_SYMBOL(lantern_addcdiv_)
  LOAD_SYMBOL(lantern_random_)
  LOAD_SYMBOL(lantern_random_)
  LOAD_SYMBOL(lantern_random_)
  LOAD_SYMBOL(lantern_uniform_)
  LOAD_SYMBOL(lantern_normal_)
  LOAD_SYMBOL(lantern_cauchy_)
  LOAD_SYMBOL(lantern_log_normal_)
  LOAD_SYMBOL(lantern_exponential_)
  LOAD_SYMBOL(lantern_geometric_)
  LOAD_SYMBOL(lantern_diag_out)
  LOAD_SYMBOL(lantern_diag)
  LOAD_SYMBOL(lantern_cross_out)
  LOAD_SYMBOL(lantern_cross)
  LOAD_SYMBOL(lantern_triu_out)
  LOAD_SYMBOL(lantern_triu)
  LOAD_SYMBOL(lantern_tril_out)
  LOAD_SYMBOL(lantern_tril)
  LOAD_SYMBOL(lantern_tril_indices)
  LOAD_SYMBOL(lantern_triu_indices)
  LOAD_SYMBOL(lantern_trace)
  LOAD_SYMBOL(lantern_ne_out)
  LOAD_SYMBOL(lantern_ne)
  LOAD_SYMBOL(lantern_ne_out)
  LOAD_SYMBOL(lantern_ne)
  LOAD_SYMBOL(lantern_eq_out)
  LOAD_SYMBOL(lantern_eq)
  LOAD_SYMBOL(lantern_eq_out)
  LOAD_SYMBOL(lantern_eq)
  LOAD_SYMBOL(lantern_ge_out)
  LOAD_SYMBOL(lantern_ge)
  LOAD_SYMBOL(lantern_ge_out)
  LOAD_SYMBOL(lantern_ge)
  LOAD_SYMBOL(lantern_le_out)
  LOAD_SYMBOL(lantern_le)
  LOAD_SYMBOL(lantern_le_out)
  LOAD_SYMBOL(lantern_le)
  LOAD_SYMBOL(lantern_gt_out)
  LOAD_SYMBOL(lantern_gt)
  LOAD_SYMBOL(lantern_gt_out)
  LOAD_SYMBOL(lantern_gt)
  LOAD_SYMBOL(lantern_lt_out)
  LOAD_SYMBOL(lantern_lt)
  LOAD_SYMBOL(lantern_lt_out)
  LOAD_SYMBOL(lantern_lt)
  LOAD_SYMBOL(lantern_take_out)
  LOAD_SYMBOL(lantern_take)
  LOAD_SYMBOL(lantern_index_select_out)
  LOAD_SYMBOL(lantern_index_select)
  LOAD_SYMBOL(lantern_index_select_out)
  LOAD_SYMBOL(lantern_index_select)
  LOAD_SYMBOL(lantern_masked_select_out)
  LOAD_SYMBOL(lantern_masked_select)
  LOAD_SYMBOL(lantern_nonzero_out)
  LOAD_SYMBOL(lantern_nonzero)
  LOAD_SYMBOL(lantern_nonzero_numpy)
  LOAD_SYMBOL(lantern_gather_out)
  LOAD_SYMBOL(lantern_gather)
  LOAD_SYMBOL(lantern_gather_out)
  LOAD_SYMBOL(lantern_gather)
  LOAD_SYMBOL(lantern__gather_sparse_backward)
  LOAD_SYMBOL(lantern_addcmul_out)
  LOAD_SYMBOL(lantern_addcmul)
  LOAD_SYMBOL(lantern_addcmul_)
  LOAD_SYMBOL(lantern_addcdiv_out)
  LOAD_SYMBOL(lantern_addcdiv)
  LOAD_SYMBOL(lantern_lstsq_out)
  LOAD_SYMBOL(lantern_lstsq)
  LOAD_SYMBOL(lantern_triangular_solve_out)
  LOAD_SYMBOL(lantern_triangular_solve)
  LOAD_SYMBOL(lantern__triangular_solve_helper)
  LOAD_SYMBOL(lantern_symeig_out)
  LOAD_SYMBOL(lantern_symeig)
  LOAD_SYMBOL(lantern__symeig_helper)
  LOAD_SYMBOL(lantern_eig_out)
  LOAD_SYMBOL(lantern_eig)
  LOAD_SYMBOL(lantern_svd_out)
  LOAD_SYMBOL(lantern_svd)
  LOAD_SYMBOL(lantern__svd_helper)
  LOAD_SYMBOL(lantern_cholesky_out)
  LOAD_SYMBOL(lantern_cholesky)
  LOAD_SYMBOL(lantern__cholesky_helper)
  LOAD_SYMBOL(lantern_cholesky_solve_out)
  LOAD_SYMBOL(lantern_cholesky_solve)
  LOAD_SYMBOL(lantern__cholesky_solve_helper)
  LOAD_SYMBOL(lantern_solve)
  LOAD_SYMBOL(lantern_solve_out)
  LOAD_SYMBOL(lantern__solve_helper)
  LOAD_SYMBOL(lantern_cholesky_inverse_out)
  LOAD_SYMBOL(lantern_cholesky_inverse)
  LOAD_SYMBOL(lantern_qr_out)
  LOAD_SYMBOL(lantern_qr)
  LOAD_SYMBOL(lantern__qr_helper)
  LOAD_SYMBOL(lantern_geqrf_out)
  LOAD_SYMBOL(lantern_geqrf)
  LOAD_SYMBOL(lantern_orgqr_out)
  LOAD_SYMBOL(lantern_orgqr)
  LOAD_SYMBOL(lantern_ormqr_out)
  LOAD_SYMBOL(lantern_ormqr)
  LOAD_SYMBOL(lantern__lu_with_info)
  LOAD_SYMBOL(lantern_lu_solve_out)
  LOAD_SYMBOL(lantern_lu_solve)
  LOAD_SYMBOL(lantern__lu_solve_helper)
  LOAD_SYMBOL(lantern_multinomial_out)
  LOAD_SYMBOL(lantern_multinomial)
  LOAD_SYMBOL(lantern__multinomial_alias_setup)
  LOAD_SYMBOL(lantern__multinomial_alias_draw)
  LOAD_SYMBOL(lantern_lgamma_out)
  LOAD_SYMBOL(lantern_lgamma)
  LOAD_SYMBOL(lantern_digamma_out)
  LOAD_SYMBOL(lantern_digamma)
  LOAD_SYMBOL(lantern_polygamma_out)
  LOAD_SYMBOL(lantern_polygamma)
  LOAD_SYMBOL(lantern_erfinv)
  LOAD_SYMBOL(lantern_erfinv_)
  LOAD_SYMBOL(lantern_erfinv_out)
  LOAD_SYMBOL(lantern_sign)
  LOAD_SYMBOL(lantern_sign_)
  LOAD_SYMBOL(lantern_sign_out)
  LOAD_SYMBOL(lantern_dist)
  LOAD_SYMBOL(lantern_atan2_out)
  LOAD_SYMBOL(lantern_atan2)
  LOAD_SYMBOL(lantern_lerp_out)
  LOAD_SYMBOL(lantern_lerp_out)
  LOAD_SYMBOL(lantern_lerp)
  LOAD_SYMBOL(lantern_lerp)
  LOAD_SYMBOL(lantern_histc_out)
  LOAD_SYMBOL(lantern_histc)
  LOAD_SYMBOL(lantern_fmod_out)
  LOAD_SYMBOL(lantern_fmod)
  LOAD_SYMBOL(lantern_fmod_out)
  LOAD_SYMBOL(lantern_fmod)
  LOAD_SYMBOL(lantern_remainder_out)
  LOAD_SYMBOL(lantern_remainder)
  LOAD_SYMBOL(lantern_remainder_out)
  LOAD_SYMBOL(lantern_remainder)
  LOAD_SYMBOL(lantern_min_out)
  LOAD_SYMBOL(lantern_min)
  LOAD_SYMBOL(lantern_min)
  LOAD_SYMBOL(lantern_max_out)
  LOAD_SYMBOL(lantern_max)
  LOAD_SYMBOL(lantern_max)
  LOAD_SYMBOL(lantern_median)
  LOAD_SYMBOL(lantern_sort_out)
  LOAD_SYMBOL(lantern_sort)
  LOAD_SYMBOL(lantern_sort_out)
  LOAD_SYMBOL(lantern_sort)
  LOAD_SYMBOL(lantern_argsort)
  LOAD_SYMBOL(lantern_argsort)
  LOAD_SYMBOL(lantern_topk_out)
  LOAD_SYMBOL(lantern_topk)
  LOAD_SYMBOL(lantern_all)
  LOAD_SYMBOL(lantern_any)
  LOAD_SYMBOL(lantern_renorm_out)
  LOAD_SYMBOL(lantern_renorm)
  LOAD_SYMBOL(lantern_unfold)
  LOAD_SYMBOL(lantern_equal)
  LOAD_SYMBOL(lantern_pow_out)
  LOAD_SYMBOL(lantern_pow)
  LOAD_SYMBOL(lantern_pow_out)
  LOAD_SYMBOL(lantern_pow)
  LOAD_SYMBOL(lantern_normal_out)
  LOAD_SYMBOL(lantern_normal)
  LOAD_SYMBOL(lantern_normal_out)
  LOAD_SYMBOL(lantern_normal)
  LOAD_SYMBOL(lantern_normal_out)
  LOAD_SYMBOL(lantern_normal)
  LOAD_SYMBOL(lantern_normal)
  LOAD_SYMBOL(lantern_normal_out)
  LOAD_SYMBOL(lantern_alias)
  LOAD_SYMBOL(lantern__addr)
  LOAD_SYMBOL(lantern__addr_)
  LOAD_SYMBOL(lantern__addr_out)
  LOAD_SYMBOL(lantern__index_copy_)
  LOAD_SYMBOL(lantern__cumsum)
  LOAD_SYMBOL(lantern__cumsum_out)
  LOAD_SYMBOL(lantern__cumprod)
  LOAD_SYMBOL(lantern__cumprod_out)
  LOAD_SYMBOL(lantern__var)
  LOAD_SYMBOL(lantern__std)
  LOAD_SYMBOL(lantern__cat)
  LOAD_SYMBOL(lantern__cat_out)
  LOAD_SYMBOL(lantern__mode)
  LOAD_SYMBOL(lantern__mode_out)
  LOAD_SYMBOL(lantern__max)
  LOAD_SYMBOL(lantern__max_out)
  LOAD_SYMBOL(lantern__min)
  LOAD_SYMBOL(lantern__min_out)
  LOAD_SYMBOL(lantern_binary_cross_entropy_out)
  LOAD_SYMBOL(lantern_binary_cross_entropy)
  LOAD_SYMBOL(lantern_binary_cross_entropy_backward_out)
  LOAD_SYMBOL(lantern_binary_cross_entropy_backward)
  LOAD_SYMBOL(lantern_mse_loss_out)
  LOAD_SYMBOL(lantern_mse_loss)
  LOAD_SYMBOL(lantern_mse_loss_backward_out)
  LOAD_SYMBOL(lantern_mse_loss_backward)
  LOAD_SYMBOL(lantern_l1_loss_out)
  LOAD_SYMBOL(lantern_l1_loss)
  LOAD_SYMBOL(lantern_l1_loss_backward_out)
  LOAD_SYMBOL(lantern_l1_loss_backward)
  LOAD_SYMBOL(lantern_multi_margin_loss_out)
  LOAD_SYMBOL(lantern_multi_margin_loss)
  LOAD_SYMBOL(lantern_multi_margin_loss_backward_out)
  LOAD_SYMBOL(lantern_multi_margin_loss_backward)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_out)
  LOAD_SYMBOL(lantern_multilabel_margin_loss)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_forward_out)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_forward)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_backward_out)
  LOAD_SYMBOL(lantern_multilabel_margin_loss_backward)
  LOAD_SYMBOL(lantern_nll_loss_out)
  LOAD_SYMBOL(lantern_nll_loss)
  LOAD_SYMBOL(lantern_nll_loss_forward_out)
  LOAD_SYMBOL(lantern_nll_loss_forward)
  LOAD_SYMBOL(lantern_nll_loss_backward_out)
  LOAD_SYMBOL(lantern_nll_loss_backward)
  LOAD_SYMBOL(lantern_nll_loss2d_out)
  LOAD_SYMBOL(lantern_nll_loss2d)
  LOAD_SYMBOL(lantern_nll_loss2d_forward_out)
  LOAD_SYMBOL(lantern_nll_loss2d_forward)
  LOAD_SYMBOL(lantern_nll_loss2d_backward_out)
  LOAD_SYMBOL(lantern_nll_loss2d_backward)
  LOAD_SYMBOL(lantern_smooth_l1_loss_out)
  LOAD_SYMBOL(lantern_smooth_l1_loss)
  LOAD_SYMBOL(lantern_smooth_l1_loss_backward_out)
  LOAD_SYMBOL(lantern_smooth_l1_loss_backward)
  LOAD_SYMBOL(lantern_soft_margin_loss_out)
  LOAD_SYMBOL(lantern_soft_margin_loss)
  LOAD_SYMBOL(lantern_soft_margin_loss_backward_out)
  LOAD_SYMBOL(lantern_soft_margin_loss_backward)
  LOAD_SYMBOL(lantern_elu_out)
  LOAD_SYMBOL(lantern_elu)
  LOAD_SYMBOL(lantern_elu_backward_out)
  LOAD_SYMBOL(lantern_elu_backward)
  LOAD_SYMBOL(lantern_elu_)
  LOAD_SYMBOL(lantern_glu_out)
  LOAD_SYMBOL(lantern_glu)
  LOAD_SYMBOL(lantern_glu_backward_out)
  LOAD_SYMBOL(lantern_glu_backward)
  LOAD_SYMBOL(lantern_hardtanh_out)
  LOAD_SYMBOL(lantern_hardtanh)
  LOAD_SYMBOL(lantern_hardtanh_backward_out)
  LOAD_SYMBOL(lantern_hardtanh_backward)
  LOAD_SYMBOL(lantern_hardtanh_)
  LOAD_SYMBOL(lantern_leaky_relu_out)
  LOAD_SYMBOL(lantern_leaky_relu)
  LOAD_SYMBOL(lantern_leaky_relu_backward_out)
  LOAD_SYMBOL(lantern_leaky_relu_backward)
  LOAD_SYMBOL(lantern_leaky_relu_)
  LOAD_SYMBOL(lantern_log_sigmoid_out)
  LOAD_SYMBOL(lantern_log_sigmoid)
  LOAD_SYMBOL(lantern_log_sigmoid_forward_out)
  LOAD_SYMBOL(lantern_log_sigmoid_forward)
  LOAD_SYMBOL(lantern_log_sigmoid_backward_out)
  LOAD_SYMBOL(lantern_log_sigmoid_backward)
  LOAD_SYMBOL(lantern_rrelu_with_noise_out)
  LOAD_SYMBOL(lantern_rrelu_with_noise)
  LOAD_SYMBOL(lantern_rrelu_with_noise_backward_out)
  LOAD_SYMBOL(lantern_rrelu_with_noise_backward)
  LOAD_SYMBOL(lantern_rrelu_with_noise_)
  LOAD_SYMBOL(lantern_softplus_out)
  LOAD_SYMBOL(lantern_softplus)
  LOAD_SYMBOL(lantern_softplus_backward_out)
  LOAD_SYMBOL(lantern_softplus_backward)
  LOAD_SYMBOL(lantern_softshrink_out)
  LOAD_SYMBOL(lantern_softshrink)
  LOAD_SYMBOL(lantern_softshrink_backward_out)
  LOAD_SYMBOL(lantern_softshrink_backward)
  LOAD_SYMBOL(lantern_adaptive_avg_pool2d_out)
  LOAD_SYMBOL(lantern_adaptive_avg_pool2d)
  LOAD_SYMBOL(lantern_mkldnn_adaptive_avg_pool2d)
  LOAD_SYMBOL(lantern__adaptive_avg_pool2d)
  LOAD_SYMBOL(lantern__adaptive_avg_pool2d_backward)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_out)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_backward_out)
  LOAD_SYMBOL(lantern_adaptive_avg_pool3d_backward)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_out)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_backward_out)
  LOAD_SYMBOL(lantern_adaptive_max_pool2d_backward)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_out)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_backward_out)
  LOAD_SYMBOL(lantern_adaptive_max_pool3d_backward)
  LOAD_SYMBOL(lantern_avg_pool2d_out)
  LOAD_SYMBOL(lantern_avg_pool2d)
  LOAD_SYMBOL(lantern_avg_pool2d_backward_out)
  LOAD_SYMBOL(lantern_avg_pool2d_backward)
  LOAD_SYMBOL(lantern_avg_pool3d_out)
  LOAD_SYMBOL(lantern_avg_pool3d)
  LOAD_SYMBOL(lantern_avg_pool3d_backward_out)
  LOAD_SYMBOL(lantern_avg_pool3d_backward)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_out)
  LOAD_SYMBOL(lantern_fractional_max_pool2d)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_backward_out)
  LOAD_SYMBOL(lantern_fractional_max_pool2d_backward)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_out)
  LOAD_SYMBOL(lantern_fractional_max_pool3d)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_backward_out)
  LOAD_SYMBOL(lantern_fractional_max_pool3d_backward)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_out)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_backward_out)
  LOAD_SYMBOL(lantern_max_pool2d_with_indices_backward)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_out)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_backward_out)
  LOAD_SYMBOL(lantern_max_pool3d_with_indices_backward)
  LOAD_SYMBOL(lantern_max_unpool2d_out)
  LOAD_SYMBOL(lantern_max_unpool2d)
  LOAD_SYMBOL(lantern_max_unpool2d_backward_out)
  LOAD_SYMBOL(lantern_max_unpool2d_backward)
  LOAD_SYMBOL(lantern_max_unpool3d_out)
  LOAD_SYMBOL(lantern_max_unpool3d)
  LOAD_SYMBOL(lantern_max_unpool3d_backward_out)
  LOAD_SYMBOL(lantern_max_unpool3d_backward)
  LOAD_SYMBOL(lantern_reflection_pad1d_out)
  LOAD_SYMBOL(lantern_reflection_pad1d)
  LOAD_SYMBOL(lantern_reflection_pad1d_backward_out)
  LOAD_SYMBOL(lantern_reflection_pad1d_backward)
  LOAD_SYMBOL(lantern_reflection_pad2d_out)
  LOAD_SYMBOL(lantern_reflection_pad2d)
  LOAD_SYMBOL(lantern_reflection_pad2d_backward_out)
  LOAD_SYMBOL(lantern_reflection_pad2d_backward)
  LOAD_SYMBOL(lantern_replication_pad1d_out)
  LOAD_SYMBOL(lantern_replication_pad1d)
  LOAD_SYMBOL(lantern_replication_pad1d_backward_out)
  LOAD_SYMBOL(lantern_replication_pad1d_backward)
  LOAD_SYMBOL(lantern_replication_pad2d_out)
  LOAD_SYMBOL(lantern_replication_pad2d)
  LOAD_SYMBOL(lantern_replication_pad2d_backward_out)
  LOAD_SYMBOL(lantern_replication_pad2d_backward)
  LOAD_SYMBOL(lantern_replication_pad3d_out)
  LOAD_SYMBOL(lantern_replication_pad3d)
  LOAD_SYMBOL(lantern_replication_pad3d_backward_out)
  LOAD_SYMBOL(lantern_replication_pad3d_backward)
  LOAD_SYMBOL(lantern_upsample_linear1d_out)
  LOAD_SYMBOL(lantern_upsample_linear1d)
  LOAD_SYMBOL(lantern_upsample_linear1d_backward_out)
  LOAD_SYMBOL(lantern_upsample_linear1d_backward)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_out)
  LOAD_SYMBOL(lantern_upsample_bilinear2d)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_backward_out)
  LOAD_SYMBOL(lantern_upsample_bilinear2d_backward)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_out)
  LOAD_SYMBOL(lantern_upsample_bicubic2d)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_backward_out)
  LOAD_SYMBOL(lantern_upsample_bicubic2d_backward)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_out)
  LOAD_SYMBOL(lantern_upsample_trilinear3d)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_backward_out)
  LOAD_SYMBOL(lantern_upsample_trilinear3d_backward)
  LOAD_SYMBOL(lantern_upsample_nearest1d_out)
  LOAD_SYMBOL(lantern_upsample_nearest1d)
  LOAD_SYMBOL(lantern_upsample_nearest1d_backward_out)
  LOAD_SYMBOL(lantern_upsample_nearest1d_backward)
  LOAD_SYMBOL(lantern_upsample_nearest2d_out)
  LOAD_SYMBOL(lantern_upsample_nearest2d)
  LOAD_SYMBOL(lantern_upsample_nearest2d_backward_out)
  LOAD_SYMBOL(lantern_upsample_nearest2d_backward)
  LOAD_SYMBOL(lantern_upsample_nearest3d_out)
  LOAD_SYMBOL(lantern_upsample_nearest3d)
  LOAD_SYMBOL(lantern_upsample_nearest3d_backward_out)
  LOAD_SYMBOL(lantern_upsample_nearest3d_backward)
  LOAD_SYMBOL(lantern_sigmoid_backward_out)
  LOAD_SYMBOL(lantern_sigmoid_backward)
  LOAD_SYMBOL(lantern_tanh_backward_out)
  LOAD_SYMBOL(lantern_tanh_backward)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_out)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_backward_out)
  LOAD_SYMBOL(lantern_slow_conv_transpose2d_backward)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_out)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_backward_out)
  LOAD_SYMBOL(lantern_slow_conv_transpose3d_backward)
  LOAD_SYMBOL(lantern_thnn_conv2d_out)
  LOAD_SYMBOL(lantern_thnn_conv2d)
  LOAD_SYMBOL(lantern_thnn_conv2d_forward_out)
  LOAD_SYMBOL(lantern_thnn_conv2d_forward)
  LOAD_SYMBOL(lantern_thnn_conv2d_backward_out)
  LOAD_SYMBOL(lantern_thnn_conv2d_backward)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_out)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_forward_out)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_forward)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_backward_out)
  LOAD_SYMBOL(lantern_thnn_conv_depthwise2d_backward)
  LOAD_SYMBOL(lantern_thnn_conv3d_out)
  LOAD_SYMBOL(lantern_thnn_conv3d)
  LOAD_SYMBOL(lantern_thnn_conv3d_forward_out)
  LOAD_SYMBOL(lantern_thnn_conv3d_forward)
  LOAD_SYMBOL(lantern_thnn_conv3d_backward_out)
  LOAD_SYMBOL(lantern_thnn_conv3d_backward)
  LOAD_SYMBOL(lantern_slow_conv_dilated2d)
  LOAD_SYMBOL(lantern_slow_conv_dilated2d_backward)
  LOAD_SYMBOL(lantern_slow_conv_dilated3d)
  LOAD_SYMBOL(lantern_slow_conv_dilated3d_backward)
  LOAD_SYMBOL(lantern_col2im_out)
  LOAD_SYMBOL(lantern_col2im)
  LOAD_SYMBOL(lantern_col2im_backward_out)
  LOAD_SYMBOL(lantern_col2im_backward)
  LOAD_SYMBOL(lantern_im2col_out)
  LOAD_SYMBOL(lantern_im2col)
  LOAD_SYMBOL(lantern_im2col_backward_out)
  LOAD_SYMBOL(lantern_im2col_backward)
  */
  /* Autogen Symbols -- End */
  
  return true;
}

#endif
#endif
