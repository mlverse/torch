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
LANTERN_API void (LANTERN_PTR _cast_Byte)();
LANTERN_API void (LANTERN_PTR _cast_Char)();
LANTERN_API void (LANTERN_PTR _cast_Double)();
LANTERN_API void (LANTERN_PTR _cast_Float)();
LANTERN_API void (LANTERN_PTR _cast_Int)();
LANTERN_API void (LANTERN_PTR _cast_Long)();
LANTERN_API void (LANTERN_PTR _cast_Short)();
LANTERN_API void (LANTERN_PTR _cast_Half)();
LANTERN_API void (LANTERN_PTR backward)();
LANTERN_API void (LANTERN_PTR set_data)();
LANTERN_API void (LANTERN_PTR data)();
LANTERN_API void (LANTERN_PTR is_leaf)();
LANTERN_API void (LANTERN_PTR output_nr)();
LANTERN_API void (LANTERN_PTR _version)();
LANTERN_API void (LANTERN_PTR rename_)();
LANTERN_API void (LANTERN_PTR rename)();
LANTERN_API void (LANTERN_PTR align_to)();
LANTERN_API void (LANTERN_PTR align_as)();
LANTERN_API void (LANTERN_PTR align_tensors)();
LANTERN_API void (LANTERN_PTR refine_names)();
LANTERN_API void (LANTERN_PTR unflatten)();
LANTERN_API void (LANTERN_PTR unflatten)();
LANTERN_API void (LANTERN_PTR _cudnn_ctc_loss)();
LANTERN_API void (LANTERN_PTR _cudnn_rnn_flatten_weight)();
LANTERN_API void (LANTERN_PTR _cudnn_rnn)();
LANTERN_API void (LANTERN_PTR _cudnn_rnn_backward)();
LANTERN_API void (LANTERN_PTR _cudnn_init_dropout_state)();
LANTERN_API void (LANTERN_PTR _debug_has_internal_overlap)();
LANTERN_API void (LANTERN_PTR _fused_dropout)();
LANTERN_API void (LANTERN_PTR _masked_scale)();
LANTERN_API void (LANTERN_PTR _sobol_engine_draw)();
LANTERN_API void (LANTERN_PTR _sobol_engine_ff_)();
LANTERN_API void (LANTERN_PTR _sobol_engine_scramble_)();
LANTERN_API void (LANTERN_PTR _sobol_engine_initialize_state_)();
LANTERN_API void (LANTERN_PTR _reshape_from_tensor)();
LANTERN_API void (LANTERN_PTR _shape_as_tensor)();
LANTERN_API void (LANTERN_PTR dropout)();
LANTERN_API void (LANTERN_PTR dropout_)();
LANTERN_API void (LANTERN_PTR feature_dropout)();
LANTERN_API void (LANTERN_PTR feature_dropout_)();
LANTERN_API void (LANTERN_PTR alpha_dropout)();
LANTERN_API void (LANTERN_PTR alpha_dropout_)();
LANTERN_API void (LANTERN_PTR feature_alpha_dropout)();
LANTERN_API void (LANTERN_PTR feature_alpha_dropout_)();
LANTERN_API void (LANTERN_PTR abs)();
LANTERN_API void (LANTERN_PTR abs_)();
LANTERN_API void (LANTERN_PTR abs_out)();
LANTERN_API void (LANTERN_PTR acos)();
LANTERN_API void (LANTERN_PTR acos_)();
LANTERN_API void (LANTERN_PTR acos_out)();
LANTERN_API void (LANTERN_PTR avg_pool1d)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool1d)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool1d)();
LANTERN_API void (LANTERN_PTR add)();
LANTERN_API void (LANTERN_PTR add_)();
LANTERN_API void (LANTERN_PTR add_out)();
LANTERN_API void (LANTERN_PTR add)();
LANTERN_API void (LANTERN_PTR add_)();
LANTERN_API void (LANTERN_PTR addmv)();
LANTERN_API void (LANTERN_PTR addmv_)();
LANTERN_API void (LANTERN_PTR addmv_out)();
LANTERN_API void (LANTERN_PTR addr)();
LANTERN_API void (LANTERN_PTR addr_)();
LANTERN_API void (LANTERN_PTR addr_out)();
LANTERN_API void (LANTERN_PTR affine_grid_generator)();
LANTERN_API void (LANTERN_PTR affine_grid_generator_backward)();
LANTERN_API void (LANTERN_PTR all)();
LANTERN_API void (LANTERN_PTR all_out)();
LANTERN_API void (LANTERN_PTR all)();
LANTERN_API void (LANTERN_PTR all_out)();
LANTERN_API void (LANTERN_PTR allclose)();
LANTERN_API void (LANTERN_PTR any)();
LANTERN_API void (LANTERN_PTR any_out)();
LANTERN_API void (LANTERN_PTR any)();
LANTERN_API void (LANTERN_PTR any_out)();
LANTERN_API void (LANTERN_PTR arange)();
LANTERN_API void (LANTERN_PTR arange)();
LANTERN_API void (LANTERN_PTR arange)();
LANTERN_API void (LANTERN_PTR arange_out)();
LANTERN_API void (LANTERN_PTR arange_out)();
LANTERN_API void (LANTERN_PTR _dim_arange)();
LANTERN_API void (LANTERN_PTR argmax)();
LANTERN_API void (LANTERN_PTR argmin)();
LANTERN_API void (LANTERN_PTR as_strided)();
LANTERN_API void (LANTERN_PTR as_strided_)();
LANTERN_API void (LANTERN_PTR asin)();
LANTERN_API void (LANTERN_PTR asin_)();
LANTERN_API void (LANTERN_PTR asin_out)();
LANTERN_API void (LANTERN_PTR atan)();
LANTERN_API void (LANTERN_PTR atan_)();
LANTERN_API void (LANTERN_PTR atan_out)();
LANTERN_API void (LANTERN_PTR baddbmm)();
LANTERN_API void (LANTERN_PTR baddbmm_)();
LANTERN_API void (LANTERN_PTR _baddbmm_mkl_)();
LANTERN_API void (LANTERN_PTR baddbmm_out)();
LANTERN_API void (LANTERN_PTR bartlett_window)();
LANTERN_API void (LANTERN_PTR bartlett_window)();
LANTERN_API void (LANTERN_PTR batch_norm)();
LANTERN_API void (LANTERN_PTR _batch_norm_impl_index)();
LANTERN_API void (LANTERN_PTR _batch_norm_impl_index_backward)();
LANTERN_API void (LANTERN_PTR bernoulli)();
LANTERN_API void (LANTERN_PTR bernoulli_out)();
LANTERN_API void (LANTERN_PTR bernoulli_)();
LANTERN_API void (LANTERN_PTR bernoulli_)();
LANTERN_API void (LANTERN_PTR bernoulli)();
LANTERN_API void (LANTERN_PTR bilinear)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy_with_logits)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy_with_logits_backward)();
LANTERN_API void (LANTERN_PTR bincount)();
LANTERN_API void (LANTERN_PTR bitwise_not)();
LANTERN_API void (LANTERN_PTR bitwise_not_)();
LANTERN_API void (LANTERN_PTR bitwise_not_out)();
LANTERN_API void (LANTERN_PTR logical_not)();
LANTERN_API void (LANTERN_PTR logical_not_)();
LANTERN_API void (LANTERN_PTR logical_not_out)();
LANTERN_API void (LANTERN_PTR logical_xor)();
LANTERN_API void (LANTERN_PTR logical_xor_)();
LANTERN_API void (LANTERN_PTR logical_xor_out)();
LANTERN_API void (LANTERN_PTR blackman_window)();
LANTERN_API void (LANTERN_PTR blackman_window)();
LANTERN_API void (LANTERN_PTR bmm)();
LANTERN_API void (LANTERN_PTR bmm_out)();
LANTERN_API void (LANTERN_PTR broadcast_tensors)();
LANTERN_API void (LANTERN_PTR cat)();
LANTERN_API void (LANTERN_PTR cat_out)();
LANTERN_API void (LANTERN_PTR cat)();
LANTERN_API void (LANTERN_PTR cat_out)();
LANTERN_API void (LANTERN_PTR ceil)();
LANTERN_API void (LANTERN_PTR ceil_)();
LANTERN_API void (LANTERN_PTR ceil_out)();
LANTERN_API void (LANTERN_PTR chain_matmul)();
LANTERN_API void (LANTERN_PTR chunk)();
LANTERN_API void (LANTERN_PTR clamp)();
LANTERN_API void (LANTERN_PTR clamp_)();
LANTERN_API void (LANTERN_PTR clamp_out)();
LANTERN_API void (LANTERN_PTR clamp_max)();
LANTERN_API void (LANTERN_PTR clamp_max_)();
LANTERN_API void (LANTERN_PTR clamp_max_out)();
LANTERN_API void (LANTERN_PTR clamp_min)();
LANTERN_API void (LANTERN_PTR clamp_min_)();
LANTERN_API void (LANTERN_PTR clamp_min_out)();
LANTERN_API void (LANTERN_PTR cudnn_is_acceptable)();
LANTERN_API void (LANTERN_PTR constant_pad_nd)();
LANTERN_API void (LANTERN_PTR contiguous)();
LANTERN_API void (LANTERN_PTR convolution)();
LANTERN_API void (LANTERN_PTR convolution_overrideable)();
LANTERN_API void (LANTERN_PTR convolution_backward_overrideable)();
LANTERN_API void (LANTERN_PTR _convolution)();
LANTERN_API void (LANTERN_PTR _convolution_nogroup)();
LANTERN_API void (LANTERN_PTR _convolution_double_backward)();
LANTERN_API void (LANTERN_PTR conv1d)();
LANTERN_API void (LANTERN_PTR conv2d)();
LANTERN_API void (LANTERN_PTR conv3d)();
LANTERN_API void (LANTERN_PTR conv_tbc)();
LANTERN_API void (LANTERN_PTR conv_tbc_backward)();
LANTERN_API void (LANTERN_PTR conv_transpose1d)();
LANTERN_API void (LANTERN_PTR conv_transpose2d)();
LANTERN_API void (LANTERN_PTR conv_transpose3d)();
LANTERN_API void (LANTERN_PTR copy_)();
LANTERN_API void (LANTERN_PTR _copy_from)();
LANTERN_API void (LANTERN_PTR cos)();
LANTERN_API void (LANTERN_PTR cos_)();
LANTERN_API void (LANTERN_PTR cos_out)();
LANTERN_API void (LANTERN_PTR cosh)();
LANTERN_API void (LANTERN_PTR cosh_)();
LANTERN_API void (LANTERN_PTR cosh_out)();
LANTERN_API void (LANTERN_PTR cosine_embedding_loss)();
LANTERN_API void (LANTERN_PTR cudnn_affine_grid_generator)();
LANTERN_API void (LANTERN_PTR cudnn_affine_grid_generator_backward)();
LANTERN_API void (LANTERN_PTR cudnn_batch_norm)();
LANTERN_API void (LANTERN_PTR cudnn_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR cudnn_convolution)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_backward)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_backward_bias)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_transpose)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_transpose_backward)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_transpose_backward_bias)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_transpose_backward_input)();
LANTERN_API void (LANTERN_PTR cudnn_convolution_transpose_backward_weight)();
LANTERN_API void (LANTERN_PTR cudnn_grid_sampler)();
LANTERN_API void (LANTERN_PTR cudnn_grid_sampler_backward)();
LANTERN_API void (LANTERN_PTR cumsum)();
LANTERN_API void (LANTERN_PTR cumsum_out)();
LANTERN_API void (LANTERN_PTR cumsum)();
LANTERN_API void (LANTERN_PTR cumsum_out)();
LANTERN_API void (LANTERN_PTR cumprod)();
LANTERN_API void (LANTERN_PTR cumprod_out)();
LANTERN_API void (LANTERN_PTR cumprod)();
LANTERN_API void (LANTERN_PTR cumprod_out)();
LANTERN_API void (LANTERN_PTR ctc_loss)();
LANTERN_API void (LANTERN_PTR ctc_loss)();
LANTERN_API void (LANTERN_PTR _ctc_loss)();
LANTERN_API void (LANTERN_PTR _ctc_loss_backward)();
LANTERN_API void (LANTERN_PTR det)();
LANTERN_API void (LANTERN_PTR diag_embed)();
LANTERN_API void (LANTERN_PTR diagflat)();
LANTERN_API void (LANTERN_PTR diagonal)();
LANTERN_API void (LANTERN_PTR fill_diagonal_)();
LANTERN_API void (LANTERN_PTR div)();
LANTERN_API void (LANTERN_PTR div_)();
LANTERN_API void (LANTERN_PTR div_out)();
LANTERN_API void (LANTERN_PTR div)();
LANTERN_API void (LANTERN_PTR div_)();
LANTERN_API void (LANTERN_PTR dot)();
LANTERN_API void (LANTERN_PTR dot_out)();
LANTERN_API void (LANTERN_PTR einsum)();
LANTERN_API void (LANTERN_PTR embedding)();
LANTERN_API void (LANTERN_PTR embedding_backward)();
LANTERN_API void (LANTERN_PTR embedding_dense_backward)();
LANTERN_API void (LANTERN_PTR embedding_renorm_)();
LANTERN_API void (LANTERN_PTR embedding_sparse_backward)();
LANTERN_API void (LANTERN_PTR embedding_bag)();
LANTERN_API void (LANTERN_PTR _embedding_bag)();
LANTERN_API void (LANTERN_PTR _embedding_bag_backward)();
LANTERN_API void (LANTERN_PTR _embedding_bag_sparse_backward)();
LANTERN_API void (LANTERN_PTR _embedding_bag_dense_backward)();
LANTERN_API void (LANTERN_PTR _embedding_bag_per_sample_weights_backward)();
LANTERN_API void (LANTERN_PTR empty)();
LANTERN_API void (LANTERN_PTR empty)();
LANTERN_API void (LANTERN_PTR new_empty)();
LANTERN_API void (LANTERN_PTR new_full)();
LANTERN_API void (LANTERN_PTR _empty_affine_quantized)();
LANTERN_API void (LANTERN_PTR _empty_per_channel_affine_quantized)();
LANTERN_API void (LANTERN_PTR resize_)();
LANTERN_API void (LANTERN_PTR empty_out)();
LANTERN_API void (LANTERN_PTR empty_like)();
LANTERN_API void (LANTERN_PTR empty_like)();
LANTERN_API void (LANTERN_PTR empty_strided)();
LANTERN_API void (LANTERN_PTR erf)();
LANTERN_API void (LANTERN_PTR erf_)();
LANTERN_API void (LANTERN_PTR erf_out)();
LANTERN_API void (LANTERN_PTR erfc)();
LANTERN_API void (LANTERN_PTR erfc_)();
LANTERN_API void (LANTERN_PTR erfc_out)();
LANTERN_API void (LANTERN_PTR exp)();
LANTERN_API void (LANTERN_PTR exp_)();
LANTERN_API void (LANTERN_PTR exp_out)();
LANTERN_API void (LANTERN_PTR expm1)();
LANTERN_API void (LANTERN_PTR expm1_)();
LANTERN_API void (LANTERN_PTR expm1_out)();
LANTERN_API void (LANTERN_PTR expand)();
LANTERN_API void (LANTERN_PTR expand_as)();
LANTERN_API void (LANTERN_PTR eye)();
LANTERN_API void (LANTERN_PTR eye)();
LANTERN_API void (LANTERN_PTR eye_out)();
LANTERN_API void (LANTERN_PTR eye_out)();
LANTERN_API void (LANTERN_PTR flatten)();
LANTERN_API void (LANTERN_PTR flatten)();
LANTERN_API void (LANTERN_PTR flatten)();
LANTERN_API void (LANTERN_PTR flatten)();
LANTERN_API void (LANTERN_PTR fill_)();
LANTERN_API void (LANTERN_PTR fill_)();
LANTERN_API void (LANTERN_PTR floor)();
LANTERN_API void (LANTERN_PTR floor_)();
LANTERN_API void (LANTERN_PTR floor_out)();
LANTERN_API void (LANTERN_PTR frac)();
LANTERN_API void (LANTERN_PTR frac_)();
LANTERN_API void (LANTERN_PTR frac_out)();
LANTERN_API void (LANTERN_PTR full)();
LANTERN_API void (LANTERN_PTR full)();
LANTERN_API void (LANTERN_PTR full_out)();
LANTERN_API void (LANTERN_PTR full_like)();
LANTERN_API void (LANTERN_PTR full_like)();
LANTERN_API void (LANTERN_PTR from_file)();
LANTERN_API void (LANTERN_PTR grid_sampler)();
LANTERN_API void (LANTERN_PTR grid_sampler_2d)();
LANTERN_API void (LANTERN_PTR grid_sampler_2d_backward)();
LANTERN_API void (LANTERN_PTR grid_sampler_3d)();
LANTERN_API void (LANTERN_PTR grid_sampler_3d_backward)();
LANTERN_API void (LANTERN_PTR hann_window)();
LANTERN_API void (LANTERN_PTR hann_window)();
LANTERN_API void (LANTERN_PTR hamming_window)();
LANTERN_API void (LANTERN_PTR hamming_window)();
LANTERN_API void (LANTERN_PTR hamming_window)();
LANTERN_API void (LANTERN_PTR hamming_window)();
LANTERN_API void (LANTERN_PTR hinge_embedding_loss)();
LANTERN_API void (LANTERN_PTR ger)();
LANTERN_API void (LANTERN_PTR ger_out)();
LANTERN_API void (LANTERN_PTR group_norm)();
LANTERN_API void (LANTERN_PTR fft)();
LANTERN_API void (LANTERN_PTR ifft)();
LANTERN_API void (LANTERN_PTR rfft)();
LANTERN_API void (LANTERN_PTR irfft)();
LANTERN_API void (LANTERN_PTR _fft_with_size)();
LANTERN_API void (LANTERN_PTR _cufft_get_plan_cache_size)();
LANTERN_API void (LANTERN_PTR _cufft_get_plan_cache_max_size)();
LANTERN_API void (LANTERN_PTR _cufft_set_plan_cache_max_size)();
LANTERN_API void (LANTERN_PTR _cufft_clear_plan_cache)();
LANTERN_API void (LANTERN_PTR index)();
LANTERN_API void (LANTERN_PTR index_copy_)();
LANTERN_API void (LANTERN_PTR index_copy)();
LANTERN_API void (LANTERN_PTR index_copy_)();
LANTERN_API void (LANTERN_PTR index_copy)();
LANTERN_API void (LANTERN_PTR index_put_)();
LANTERN_API void (LANTERN_PTR index_put)();
LANTERN_API void (LANTERN_PTR _index_put_impl_)();
LANTERN_API void (LANTERN_PTR instance_norm)();
LANTERN_API void (LANTERN_PTR inverse)();
LANTERN_API void (LANTERN_PTR inverse_out)();
LANTERN_API void (LANTERN_PTR _inverse_helper)();
LANTERN_API void (LANTERN_PTR isclose)();
LANTERN_API void (LANTERN_PTR isnan)();
LANTERN_API void (LANTERN_PTR is_distributed)();
LANTERN_API void (LANTERN_PTR is_floating_point)();
LANTERN_API void (LANTERN_PTR is_complex)();
LANTERN_API void (LANTERN_PTR is_nonzero)();
LANTERN_API void (LANTERN_PTR is_same_size)();
LANTERN_API void (LANTERN_PTR is_signed)();
LANTERN_API void (LANTERN_PTR kl_div)();
LANTERN_API void (LANTERN_PTR kl_div_backward)();
LANTERN_API void (LANTERN_PTR kthvalue)();
LANTERN_API void (LANTERN_PTR kthvalue_out)();
LANTERN_API void (LANTERN_PTR kthvalue)();
LANTERN_API void (LANTERN_PTR kthvalue_out)();
LANTERN_API void (LANTERN_PTR layer_norm)();
LANTERN_API void (LANTERN_PTR native_layer_norm)();
LANTERN_API void (LANTERN_PTR native_layer_norm_backward)();
LANTERN_API void (LANTERN_PTR native_layer_norm_double_backward)();
LANTERN_API void (LANTERN_PTR linear)();
LANTERN_API void (LANTERN_PTR mkldnn_linear)();
LANTERN_API void (LANTERN_PTR fbgemm_linear_int8_weight_fp32_activation)();
LANTERN_API void (LANTERN_PTR fbgemm_linear_int8_weight)();
LANTERN_API void (LANTERN_PTR fbgemm_linear_quantize_weight)();
LANTERN_API void (LANTERN_PTR fbgemm_pack_gemm_matrix_fp16)();
LANTERN_API void (LANTERN_PTR fbgemm_linear_fp16_weight_fp32_activation)();
LANTERN_API void (LANTERN_PTR fbgemm_linear_fp16_weight)();
LANTERN_API void (LANTERN_PTR fbgemm_pack_quantized_matrix)();
LANTERN_API void (LANTERN_PTR fbgemm_pack_quantized_matrix)();
LANTERN_API void (LANTERN_PTR linspace)();
LANTERN_API void (LANTERN_PTR linspace_out)();
LANTERN_API void (LANTERN_PTR log)();
LANTERN_API void (LANTERN_PTR log_)();
LANTERN_API void (LANTERN_PTR log_out)();
LANTERN_API void (LANTERN_PTR log10)();
LANTERN_API void (LANTERN_PTR log10_)();
LANTERN_API void (LANTERN_PTR log10_out)();
LANTERN_API void (LANTERN_PTR log1p)();
LANTERN_API void (LANTERN_PTR log1p_)();
LANTERN_API void (LANTERN_PTR log1p_out)();
LANTERN_API void (LANTERN_PTR log2)();
LANTERN_API void (LANTERN_PTR log2_)();
LANTERN_API void (LANTERN_PTR log2_out)();
LANTERN_API void (LANTERN_PTR logdet)();
LANTERN_API void (LANTERN_PTR logspace)();
LANTERN_API void (LANTERN_PTR logspace_out)();
LANTERN_API void (LANTERN_PTR log_softmax)();
LANTERN_API void (LANTERN_PTR log_softmax)();
LANTERN_API void (LANTERN_PTR _log_softmax)();
LANTERN_API void (LANTERN_PTR _log_softmax_backward_data)();
LANTERN_API void (LANTERN_PTR logsumexp)();
LANTERN_API void (LANTERN_PTR logsumexp_out)();
LANTERN_API void (LANTERN_PTR logsumexp)();
LANTERN_API void (LANTERN_PTR logsumexp_out)();
LANTERN_API void (LANTERN_PTR margin_ranking_loss)();
LANTERN_API void (LANTERN_PTR matmul)();
LANTERN_API void (LANTERN_PTR matmul_out)();
LANTERN_API void (LANTERN_PTR matrix_rank)();
LANTERN_API void (LANTERN_PTR matrix_rank)();
LANTERN_API void (LANTERN_PTR matrix_power)();
LANTERN_API void (LANTERN_PTR max)();
LANTERN_API void (LANTERN_PTR max_out)();
LANTERN_API void (LANTERN_PTR max_values)();
LANTERN_API void (LANTERN_PTR max)();
LANTERN_API void (LANTERN_PTR max_out)();
LANTERN_API void (LANTERN_PTR max_values)();
LANTERN_API void (LANTERN_PTR max_pool1d_with_indices)();
LANTERN_API void (LANTERN_PTR max_pool1d)();
LANTERN_API void (LANTERN_PTR max_pool2d)();
LANTERN_API void (LANTERN_PTR mkldnn_max_pool2d)();
LANTERN_API void (LANTERN_PTR quantized_max_pool2d)();
LANTERN_API void (LANTERN_PTR max_pool3d)();
LANTERN_API void (LANTERN_PTR mean)();
LANTERN_API void (LANTERN_PTR mean)();
LANTERN_API void (LANTERN_PTR mean_out)();
LANTERN_API void (LANTERN_PTR mean)();
LANTERN_API void (LANTERN_PTR mean_out)();
LANTERN_API void (LANTERN_PTR median)();
LANTERN_API void (LANTERN_PTR median_out)();
LANTERN_API void (LANTERN_PTR median)();
LANTERN_API void (LANTERN_PTR median_out)();
LANTERN_API void (LANTERN_PTR min)();
LANTERN_API void (LANTERN_PTR min_out)();
LANTERN_API void (LANTERN_PTR min_values)();
LANTERN_API void (LANTERN_PTR min)();
LANTERN_API void (LANTERN_PTR min_out)();
LANTERN_API void (LANTERN_PTR min_values)();
LANTERN_API void (LANTERN_PTR mkldnn_convolution)();
LANTERN_API void (LANTERN_PTR mkldnn_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR mkldnn_convolution_backward_weights)();
LANTERN_API void (LANTERN_PTR mkldnn_convolution_backward)();
LANTERN_API void (LANTERN_PTR miopen_batch_norm)();
LANTERN_API void (LANTERN_PTR miopen_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR miopen_convolution)();
LANTERN_API void (LANTERN_PTR miopen_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR miopen_convolution_backward)();
LANTERN_API void (LANTERN_PTR miopen_convolution_backward_bias)();
LANTERN_API void (LANTERN_PTR miopen_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR miopen_convolution_transpose)();
LANTERN_API void (LANTERN_PTR miopen_convolution_transpose_backward)();
LANTERN_API void (LANTERN_PTR miopen_convolution_transpose_backward_input)();
LANTERN_API void (LANTERN_PTR miopen_convolution_transpose_backward_weight)();
LANTERN_API void (LANTERN_PTR miopen_depthwise_convolution)();
LANTERN_API void (LANTERN_PTR miopen_depthwise_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR miopen_depthwise_convolution_backward)();
LANTERN_API void (LANTERN_PTR miopen_depthwise_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR miopen_rnn)();
LANTERN_API void (LANTERN_PTR miopen_rnn_backward)();
LANTERN_API void (LANTERN_PTR mm)();
LANTERN_API void (LANTERN_PTR mm_out)();
LANTERN_API void (LANTERN_PTR _sparse_mm)();
LANTERN_API void (LANTERN_PTR mode)();
LANTERN_API void (LANTERN_PTR mode_out)();
LANTERN_API void (LANTERN_PTR mode)();
LANTERN_API void (LANTERN_PTR mode_out)();
LANTERN_API void (LANTERN_PTR mul)();
LANTERN_API void (LANTERN_PTR mul_)();
LANTERN_API void (LANTERN_PTR mul_out)();
LANTERN_API void (LANTERN_PTR mul)();
LANTERN_API void (LANTERN_PTR mul_)();
LANTERN_API void (LANTERN_PTR mv)();
LANTERN_API void (LANTERN_PTR mv_out)();
LANTERN_API void (LANTERN_PTR mvlgamma)();
LANTERN_API void (LANTERN_PTR mvlgamma_)();
LANTERN_API void (LANTERN_PTR narrow_copy)();
LANTERN_API void (LANTERN_PTR narrow)();
LANTERN_API void (LANTERN_PTR native_batch_norm)();
LANTERN_API void (LANTERN_PTR batch_norm_stats)();
LANTERN_API void (LANTERN_PTR batch_norm_elemt)();
LANTERN_API void (LANTERN_PTR batch_norm_gather_stats)();
LANTERN_API void (LANTERN_PTR batch_norm_gather_stats_with_counts)();
LANTERN_API void (LANTERN_PTR native_batch_norm_backward)();
LANTERN_API void (LANTERN_PTR batch_norm_backward_reduce)();
LANTERN_API void (LANTERN_PTR batch_norm_backward_elemt)();
LANTERN_API void (LANTERN_PTR batch_norm_update_stats)();
LANTERN_API void (LANTERN_PTR _nnpack_available)();
LANTERN_API void (LANTERN_PTR _nnpack_spatial_convolution)();
LANTERN_API void (LANTERN_PTR _nnpack_spatial_convolution_backward)();
LANTERN_API void (LANTERN_PTR _nnpack_spatial_convolution_backward_input)();
LANTERN_API void (LANTERN_PTR _nnpack_spatial_convolution_backward_weight)();
LANTERN_API void (LANTERN_PTR ones)();
LANTERN_API void (LANTERN_PTR ones)();
LANTERN_API void (LANTERN_PTR ones_out)();
LANTERN_API void (LANTERN_PTR ones_like)();
LANTERN_API void (LANTERN_PTR ones_like)();
LANTERN_API void (LANTERN_PTR pairwise_distance)();
LANTERN_API void (LANTERN_PTR cdist)();
LANTERN_API void (LANTERN_PTR _cdist_backward)();
LANTERN_API void (LANTERN_PTR pdist)();
LANTERN_API void (LANTERN_PTR _pdist_forward)();
LANTERN_API void (LANTERN_PTR _pdist_backward)();
LANTERN_API void (LANTERN_PTR cosine_similarity)();
LANTERN_API void (LANTERN_PTR permute)();
LANTERN_API void (LANTERN_PTR numpy_T)();
LANTERN_API void (LANTERN_PTR pixel_shuffle)();
LANTERN_API void (LANTERN_PTR is_pinned)();
LANTERN_API void (LANTERN_PTR pin_memory)();
LANTERN_API void (LANTERN_PTR pinverse)();
LANTERN_API void (LANTERN_PTR poisson_nll_loss)();
LANTERN_API void (LANTERN_PTR scalar_tensor)();
LANTERN_API void (LANTERN_PTR rand)();
LANTERN_API void (LANTERN_PTR rand)();
LANTERN_API void (LANTERN_PTR rand)();
LANTERN_API void (LANTERN_PTR rand)();
LANTERN_API void (LANTERN_PTR rand_out)();
LANTERN_API void (LANTERN_PTR rand_out)();
LANTERN_API void (LANTERN_PTR rand_like)();
LANTERN_API void (LANTERN_PTR rand_like)();
LANTERN_API void (LANTERN_PTR randint)();
LANTERN_API void (LANTERN_PTR randint)();
LANTERN_API void (LANTERN_PTR randint)();
LANTERN_API void (LANTERN_PTR randint)();
LANTERN_API void (LANTERN_PTR randint_out)();
LANTERN_API void (LANTERN_PTR randint_out)();
LANTERN_API void (LANTERN_PTR randint_out)();
LANTERN_API void (LANTERN_PTR randint_out)();
LANTERN_API void (LANTERN_PTR randint_like)();
LANTERN_API void (LANTERN_PTR randint_like)();
LANTERN_API void (LANTERN_PTR randint_like)();
LANTERN_API void (LANTERN_PTR randint_like)();
LANTERN_API void (LANTERN_PTR randn)();
LANTERN_API void (LANTERN_PTR randn)();
LANTERN_API void (LANTERN_PTR randn)();
LANTERN_API void (LANTERN_PTR randn)();
LANTERN_API void (LANTERN_PTR randn_out)();
LANTERN_API void (LANTERN_PTR randn_out)();
LANTERN_API void (LANTERN_PTR randn_like)();
LANTERN_API void (LANTERN_PTR randn_like)();
LANTERN_API void (LANTERN_PTR randperm)();
LANTERN_API void (LANTERN_PTR randperm)();
LANTERN_API void (LANTERN_PTR randperm_out)();
LANTERN_API void (LANTERN_PTR randperm_out)();
LANTERN_API void (LANTERN_PTR range)();
LANTERN_API void (LANTERN_PTR range)();
LANTERN_API void (LANTERN_PTR range_out)();
LANTERN_API void (LANTERN_PTR reciprocal)();
LANTERN_API void (LANTERN_PTR reciprocal_)();
LANTERN_API void (LANTERN_PTR reciprocal_out)();
LANTERN_API void (LANTERN_PTR neg)();
LANTERN_API void (LANTERN_PTR neg_)();
LANTERN_API void (LANTERN_PTR neg_out)();
LANTERN_API void (LANTERN_PTR repeat)();
LANTERN_API void (LANTERN_PTR repeat_interleave)();
LANTERN_API void (LANTERN_PTR repeat_interleave)();
LANTERN_API void (LANTERN_PTR repeat_interleave)();
LANTERN_API void (LANTERN_PTR reshape)();
LANTERN_API void (LANTERN_PTR _mkldnn_reshape)();
LANTERN_API void (LANTERN_PTR reshape_as)();
LANTERN_API void (LANTERN_PTR round)();
LANTERN_API void (LANTERN_PTR round_)();
LANTERN_API void (LANTERN_PTR round_out)();
LANTERN_API void (LANTERN_PTR rrelu)();
LANTERN_API void (LANTERN_PTR rrelu_)();
LANTERN_API void (LANTERN_PTR relu)();
LANTERN_API void (LANTERN_PTR relu_)();
LANTERN_API void (LANTERN_PTR prelu)();
LANTERN_API void (LANTERN_PTR prelu_backward)();
LANTERN_API void (LANTERN_PTR gelu)();
LANTERN_API void (LANTERN_PTR gelu_backward)();
LANTERN_API void (LANTERN_PTR hardshrink)();
LANTERN_API void (LANTERN_PTR hardshrink_backward)();
LANTERN_API void (LANTERN_PTR rsqrt)();
LANTERN_API void (LANTERN_PTR rsqrt_)();
LANTERN_API void (LANTERN_PTR rsqrt_out)();
LANTERN_API void (LANTERN_PTR select)();
LANTERN_API void (LANTERN_PTR select)();
LANTERN_API void (LANTERN_PTR selu)();
LANTERN_API void (LANTERN_PTR selu_)();
LANTERN_API void (LANTERN_PTR celu)();
LANTERN_API void (LANTERN_PTR celu_)();
LANTERN_API void (LANTERN_PTR sigmoid)();
LANTERN_API void (LANTERN_PTR sigmoid_)();
LANTERN_API void (LANTERN_PTR sigmoid_out)();
LANTERN_API void (LANTERN_PTR sin)();
LANTERN_API void (LANTERN_PTR sin_)();
LANTERN_API void (LANTERN_PTR sin_out)();
LANTERN_API void (LANTERN_PTR sinh)();
LANTERN_API void (LANTERN_PTR sinh_)();
LANTERN_API void (LANTERN_PTR sinh_out)();
LANTERN_API void (LANTERN_PTR detach)();
LANTERN_API void (LANTERN_PTR detach_)();
LANTERN_API void (LANTERN_PTR size)();
LANTERN_API void (LANTERN_PTR size)();
LANTERN_API void (LANTERN_PTR slice)();
LANTERN_API void (LANTERN_PTR slogdet)();
LANTERN_API void (LANTERN_PTR smm)();
LANTERN_API void (LANTERN_PTR softmax)();
LANTERN_API void (LANTERN_PTR softmax)();
LANTERN_API void (LANTERN_PTR _softmax)();
LANTERN_API void (LANTERN_PTR _softmax_backward_data)();
LANTERN_API void (LANTERN_PTR split)();
LANTERN_API void (LANTERN_PTR split_with_sizes)();
LANTERN_API void (LANTERN_PTR squeeze)();
LANTERN_API void (LANTERN_PTR squeeze)();
LANTERN_API void (LANTERN_PTR squeeze)();
LANTERN_API void (LANTERN_PTR squeeze_)();
LANTERN_API void (LANTERN_PTR squeeze_)();
LANTERN_API void (LANTERN_PTR squeeze_)();
LANTERN_API void (LANTERN_PTR sspaddmm)();
LANTERN_API void (LANTERN_PTR sspaddmm_out)();
LANTERN_API void (LANTERN_PTR stack)();
LANTERN_API void (LANTERN_PTR stack_out)();
LANTERN_API void (LANTERN_PTR stft)();
LANTERN_API void (LANTERN_PTR stride)();
LANTERN_API void (LANTERN_PTR stride)();
LANTERN_API void (LANTERN_PTR sum)();
LANTERN_API void (LANTERN_PTR sum)();
LANTERN_API void (LANTERN_PTR sum)();
LANTERN_API void (LANTERN_PTR sum_out)();
LANTERN_API void (LANTERN_PTR sum_out)();
LANTERN_API void (LANTERN_PTR sum_to_size)();
LANTERN_API void (LANTERN_PTR sqrt)();
LANTERN_API void (LANTERN_PTR sqrt_)();
LANTERN_API void (LANTERN_PTR sqrt_out)();
LANTERN_API void (LANTERN_PTR std)();
LANTERN_API void (LANTERN_PTR std)();
LANTERN_API void (LANTERN_PTR std_mean)();
LANTERN_API void (LANTERN_PTR std_mean)();
LANTERN_API void (LANTERN_PTR std_mean)();
LANTERN_API void (LANTERN_PTR std_out)();
LANTERN_API void (LANTERN_PTR std)();
LANTERN_API void (LANTERN_PTR std_out)();
LANTERN_API void (LANTERN_PTR prod)();
LANTERN_API void (LANTERN_PTR prod)();
LANTERN_API void (LANTERN_PTR prod_out)();
LANTERN_API void (LANTERN_PTR prod)();
LANTERN_API void (LANTERN_PTR prod_out)();
LANTERN_API void (LANTERN_PTR t)();
LANTERN_API void (LANTERN_PTR t_)();
LANTERN_API void (LANTERN_PTR tan)();
LANTERN_API void (LANTERN_PTR tan_)();
LANTERN_API void (LANTERN_PTR tan_out)();
LANTERN_API void (LANTERN_PTR tanh)();
LANTERN_API void (LANTERN_PTR tanh_)();
LANTERN_API void (LANTERN_PTR tanh_out)();
LANTERN_API void (LANTERN_PTR tensordot)();
LANTERN_API void (LANTERN_PTR threshold)();
LANTERN_API void (LANTERN_PTR threshold_)();
LANTERN_API void (LANTERN_PTR threshold_out)();
LANTERN_API void (LANTERN_PTR threshold_backward)();
LANTERN_API void (LANTERN_PTR transpose)();
LANTERN_API void (LANTERN_PTR transpose)();
LANTERN_API void (LANTERN_PTR _mkldnn_transpose)();
LANTERN_API void (LANTERN_PTR transpose_)();
LANTERN_API void (LANTERN_PTR _mkldnn_transpose_)();
LANTERN_API void (LANTERN_PTR one_hot)();
LANTERN_API void (LANTERN_PTR flip)();
LANTERN_API void (LANTERN_PTR roll)();
LANTERN_API void (LANTERN_PTR rot90)();
LANTERN_API void (LANTERN_PTR trapz)();
LANTERN_API void (LANTERN_PTR trapz)();
LANTERN_API void (LANTERN_PTR _trilinear)();
LANTERN_API void (LANTERN_PTR triplet_margin_loss)();
LANTERN_API void (LANTERN_PTR trunc)();
LANTERN_API void (LANTERN_PTR trunc_)();
LANTERN_API void (LANTERN_PTR trunc_out)();
LANTERN_API void (LANTERN_PTR type_as)();
LANTERN_API void (LANTERN_PTR _has_compatible_shallow_copy_type)();
LANTERN_API void (LANTERN_PTR _unique)();
LANTERN_API void (LANTERN_PTR unique_dim)();
LANTERN_API void (LANTERN_PTR unique_consecutive)();
LANTERN_API void (LANTERN_PTR unique_dim_consecutive)();
LANTERN_API void (LANTERN_PTR _unique2)();
LANTERN_API void (LANTERN_PTR _unsafe_view)();
LANTERN_API void (LANTERN_PTR unsqueeze)();
LANTERN_API void (LANTERN_PTR unsqueeze_)();
LANTERN_API void (LANTERN_PTR var)();
LANTERN_API void (LANTERN_PTR var)();
LANTERN_API void (LANTERN_PTR var_out)();
LANTERN_API void (LANTERN_PTR var)();
LANTERN_API void (LANTERN_PTR var_out)();
LANTERN_API void (LANTERN_PTR var_mean)();
LANTERN_API void (LANTERN_PTR var_mean)();
LANTERN_API void (LANTERN_PTR var_mean)();
LANTERN_API void (LANTERN_PTR view_as)();
LANTERN_API void (LANTERN_PTR where)();
LANTERN_API void (LANTERN_PTR where)();
LANTERN_API void (LANTERN_PTR _s_where)();
LANTERN_API void (LANTERN_PTR norm_except_dim)();
LANTERN_API void (LANTERN_PTR _weight_norm)();
LANTERN_API void (LANTERN_PTR _weight_norm_cuda_interface)();
LANTERN_API void (LANTERN_PTR _weight_norm_cuda_interface_backward)();
LANTERN_API void (LANTERN_PTR _weight_norm_differentiable_backward)();
LANTERN_API void (LANTERN_PTR zeros)();
LANTERN_API void (LANTERN_PTR zeros)();
LANTERN_API void (LANTERN_PTR zeros_out)();
LANTERN_API void (LANTERN_PTR zeros_like)();
LANTERN_API void (LANTERN_PTR zeros_like)();
LANTERN_API void (LANTERN_PTR _standard_gamma_grad)();
LANTERN_API void (LANTERN_PTR _standard_gamma)();
LANTERN_API void (LANTERN_PTR _dirichlet_grad)();
LANTERN_API void (LANTERN_PTR _sample_dirichlet)();
LANTERN_API void (LANTERN_PTR poisson)();
LANTERN_API void (LANTERN_PTR native_norm)();
LANTERN_API void (LANTERN_PTR _sparse_sum)();
LANTERN_API void (LANTERN_PTR _sparse_sum)();
LANTERN_API void (LANTERN_PTR _sparse_sum)();
LANTERN_API void (LANTERN_PTR _sparse_sum)();
LANTERN_API void (LANTERN_PTR _sparse_sum_backward)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm_out)();
LANTERN_API void (LANTERN_PTR norm_out)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm)();
LANTERN_API void (LANTERN_PTR norm_out)();
LANTERN_API void (LANTERN_PTR norm_out)();
LANTERN_API void (LANTERN_PTR frobenius_norm)();
LANTERN_API void (LANTERN_PTR frobenius_norm)();
LANTERN_API void (LANTERN_PTR frobenius_norm_out)();
LANTERN_API void (LANTERN_PTR nuclear_norm)();
LANTERN_API void (LANTERN_PTR nuclear_norm_out)();
LANTERN_API void (LANTERN_PTR nuclear_norm)();
LANTERN_API void (LANTERN_PTR nuclear_norm_out)();
LANTERN_API void (LANTERN_PTR clone)();
LANTERN_API void (LANTERN_PTR resize_as_)();
LANTERN_API void (LANTERN_PTR pow_out)();
LANTERN_API void (LANTERN_PTR pow)();
LANTERN_API void (LANTERN_PTR zero_)();
LANTERN_API void (LANTERN_PTR sub_out)();
LANTERN_API void (LANTERN_PTR sub)();
LANTERN_API void (LANTERN_PTR sub_)();
LANTERN_API void (LANTERN_PTR sub)();
LANTERN_API void (LANTERN_PTR sub_)();
LANTERN_API void (LANTERN_PTR rsub)();
LANTERN_API void (LANTERN_PTR rsub)();
LANTERN_API void (LANTERN_PTR _sparse_addmm)();
LANTERN_API void (LANTERN_PTR addmm_out)();
LANTERN_API void (LANTERN_PTR addmm)();
LANTERN_API void (LANTERN_PTR addmm_)();
LANTERN_API void (LANTERN_PTR sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR sparse_coo_tensor)();
LANTERN_API void (LANTERN_PTR _sparse_coo_tensor_unsafe)();
LANTERN_API void (LANTERN_PTR _sparse_coo_tensor_with_dims)();
LANTERN_API void (LANTERN_PTR _sparse_coo_tensor_with_dims_and_tensors)();
LANTERN_API void (LANTERN_PTR sparse_resize_)();
LANTERN_API void (LANTERN_PTR sparse_resize_and_clear_)();
LANTERN_API void (LANTERN_PTR sparse_mask)();
LANTERN_API void (LANTERN_PTR to_dense)();
LANTERN_API void (LANTERN_PTR to_dense_backward)();
LANTERN_API void (LANTERN_PTR sparse_dim)();
LANTERN_API void (LANTERN_PTR _dimI)();
LANTERN_API void (LANTERN_PTR dense_dim)();
LANTERN_API void (LANTERN_PTR _dimV)();
LANTERN_API void (LANTERN_PTR _nnz)();
LANTERN_API void (LANTERN_PTR coalesce)();
LANTERN_API void (LANTERN_PTR is_coalesced)();
LANTERN_API void (LANTERN_PTR _indices)();
LANTERN_API void (LANTERN_PTR _values)();
LANTERN_API void (LANTERN_PTR _coalesced_)();
LANTERN_API void (LANTERN_PTR indices)();
LANTERN_API void (LANTERN_PTR values)();
LANTERN_API void (LANTERN_PTR hspmm_out)();
LANTERN_API void (LANTERN_PTR hspmm)();
LANTERN_API void (LANTERN_PTR copy_sparse_to_sparse_)();
LANTERN_API void (LANTERN_PTR numel)();
LANTERN_API void (LANTERN_PTR unbind)();
LANTERN_API void (LANTERN_PTR unbind)();
LANTERN_API void (LANTERN_PTR to_sparse)();
LANTERN_API void (LANTERN_PTR to_sparse)();
LANTERN_API void (LANTERN_PTR to_mkldnn)();
LANTERN_API void (LANTERN_PTR mkldnn_reorder_conv2d_weight)();
LANTERN_API void (LANTERN_PTR to_mkldnn_backward)();
LANTERN_API void (LANTERN_PTR quantize_per_tensor)();
LANTERN_API void (LANTERN_PTR quantize_per_channel)();
LANTERN_API void (LANTERN_PTR dequantize)();
LANTERN_API void (LANTERN_PTR q_scale)();
LANTERN_API void (LANTERN_PTR q_zero_point)();
LANTERN_API void (LANTERN_PTR q_per_channel_scales)();
LANTERN_API void (LANTERN_PTR q_per_channel_zero_points)();
LANTERN_API void (LANTERN_PTR q_per_channel_axis)();
LANTERN_API void (LANTERN_PTR int_repr)();
LANTERN_API void (LANTERN_PTR _make_per_tensor_quantized_tensor)();
LANTERN_API void (LANTERN_PTR _make_per_channel_quantized_tensor)();
LANTERN_API void (LANTERN_PTR qscheme)();
LANTERN_API void (LANTERN_PTR fake_quantize_per_tensor_affine)();
LANTERN_API void (LANTERN_PTR fake_quantize_per_tensor_affine_backward)();
LANTERN_API void (LANTERN_PTR fake_quantize_per_channel_affine)();
LANTERN_API void (LANTERN_PTR fake_quantize_per_channel_affine_backward)();
LANTERN_API void (LANTERN_PTR to)();
LANTERN_API void (LANTERN_PTR to)();
LANTERN_API void (LANTERN_PTR to)();
LANTERN_API void (LANTERN_PTR to)();
LANTERN_API void (LANTERN_PTR meshgrid)();
LANTERN_API void (LANTERN_PTR cartesian_prod)();
LANTERN_API void (LANTERN_PTR combinations)();
LANTERN_API void (LANTERN_PTR item)();
LANTERN_API void (LANTERN_PTR result_type)();
LANTERN_API void (LANTERN_PTR result_type)();
LANTERN_API void (LANTERN_PTR result_type)();
LANTERN_API void (LANTERN_PTR result_type)();
LANTERN_API void (LANTERN_PTR can_cast)();
LANTERN_API void (LANTERN_PTR promote_types)();
LANTERN_API void (LANTERN_PTR _local_scalar_dense)();
LANTERN_API void (LANTERN_PTR _thnn_fused_lstm_cell)();
LANTERN_API void (LANTERN_PTR _thnn_fused_lstm_cell_backward)();
LANTERN_API void (LANTERN_PTR _thnn_differentiable_lstm_cell_backward)();
LANTERN_API void (LANTERN_PTR _thnn_fused_gru_cell)();
LANTERN_API void (LANTERN_PTR _thnn_fused_gru_cell_backward)();
LANTERN_API void (LANTERN_PTR _thnn_differentiable_gru_cell_backward)();
LANTERN_API void (LANTERN_PTR lstm)();
LANTERN_API void (LANTERN_PTR lstm)();
LANTERN_API void (LANTERN_PTR gru)();
LANTERN_API void (LANTERN_PTR gru)();
LANTERN_API void (LANTERN_PTR rnn_tanh)();
LANTERN_API void (LANTERN_PTR rnn_tanh)();
LANTERN_API void (LANTERN_PTR rnn_relu)();
LANTERN_API void (LANTERN_PTR rnn_relu)();
LANTERN_API void (LANTERN_PTR lstm_cell)();
LANTERN_API void (LANTERN_PTR gru_cell)();
LANTERN_API void (LANTERN_PTR rnn_tanh_cell)();
LANTERN_API void (LANTERN_PTR rnn_relu_cell)();
LANTERN_API void (LANTERN_PTR quantized_lstm)();
LANTERN_API void (LANTERN_PTR quantized_gru)();
LANTERN_API void (LANTERN_PTR quantized_gru)();
LANTERN_API void (LANTERN_PTR quantized_lstm_cell)();
LANTERN_API void (LANTERN_PTR quantized_gru_cell)();
LANTERN_API void (LANTERN_PTR quantized_rnn_relu_cell)();
LANTERN_API void (LANTERN_PTR quantized_rnn_tanh_cell)();
LANTERN_API void (LANTERN_PTR _pack_padded_sequence)();
LANTERN_API void (LANTERN_PTR _pack_padded_sequence_backward)();
LANTERN_API void (LANTERN_PTR _pad_packed_sequence)();
LANTERN_API void (LANTERN_PTR set_)();
LANTERN_API void (LANTERN_PTR set_)();
LANTERN_API void (LANTERN_PTR set_)();
LANTERN_API void (LANTERN_PTR set_)();
LANTERN_API void (LANTERN_PTR set_quantizer_)();
LANTERN_API void (LANTERN_PTR is_set_to)();
LANTERN_API void (LANTERN_PTR masked_fill_)();
LANTERN_API void (LANTERN_PTR masked_fill)();
LANTERN_API void (LANTERN_PTR masked_fill_)();
LANTERN_API void (LANTERN_PTR masked_fill)();
LANTERN_API void (LANTERN_PTR masked_scatter_)();
LANTERN_API void (LANTERN_PTR masked_scatter)();
LANTERN_API void (LANTERN_PTR view)();
LANTERN_API void (LANTERN_PTR put_)();
LANTERN_API void (LANTERN_PTR index_add_)();
LANTERN_API void (LANTERN_PTR index_add)();
LANTERN_API void (LANTERN_PTR index_add)();
LANTERN_API void (LANTERN_PTR index_fill_)();
LANTERN_API void (LANTERN_PTR index_fill)();
LANTERN_API void (LANTERN_PTR index_fill_)();
LANTERN_API void (LANTERN_PTR index_fill)();
LANTERN_API void (LANTERN_PTR index_fill_)();
LANTERN_API void (LANTERN_PTR index_fill_)();
LANTERN_API void (LANTERN_PTR index_fill)();
LANTERN_API void (LANTERN_PTR index_fill)();
LANTERN_API void (LANTERN_PTR scatter_)();
LANTERN_API void (LANTERN_PTR scatter)();
LANTERN_API void (LANTERN_PTR scatter_)();
LANTERN_API void (LANTERN_PTR scatter)();
LANTERN_API void (LANTERN_PTR scatter)();
LANTERN_API void (LANTERN_PTR scatter)();
LANTERN_API void (LANTERN_PTR scatter_add_)();
LANTERN_API void (LANTERN_PTR scatter_add)();
LANTERN_API void (LANTERN_PTR scatter_add)();
LANTERN_API void (LANTERN_PTR lt_)();
LANTERN_API void (LANTERN_PTR lt_)();
LANTERN_API void (LANTERN_PTR gt_)();
LANTERN_API void (LANTERN_PTR gt_)();
LANTERN_API void (LANTERN_PTR le_)();
LANTERN_API void (LANTERN_PTR le_)();
LANTERN_API void (LANTERN_PTR ge_)();
LANTERN_API void (LANTERN_PTR ge_)();
LANTERN_API void (LANTERN_PTR eq_)();
LANTERN_API void (LANTERN_PTR eq_)();
LANTERN_API void (LANTERN_PTR ne_)();
LANTERN_API void (LANTERN_PTR ne_)();
LANTERN_API void (LANTERN_PTR __and__)();
LANTERN_API void (LANTERN_PTR __and__)();
LANTERN_API void (LANTERN_PTR __iand__)();
LANTERN_API void (LANTERN_PTR __iand__)();
LANTERN_API void (LANTERN_PTR __or__)();
LANTERN_API void (LANTERN_PTR __or__)();
LANTERN_API void (LANTERN_PTR __ior__)();
LANTERN_API void (LANTERN_PTR __ior__)();
LANTERN_API void (LANTERN_PTR __xor__)();
LANTERN_API void (LANTERN_PTR __xor__)();
LANTERN_API void (LANTERN_PTR __ixor__)();
LANTERN_API void (LANTERN_PTR __ixor__)();
LANTERN_API void (LANTERN_PTR __lshift__)();
LANTERN_API void (LANTERN_PTR __lshift__)();
LANTERN_API void (LANTERN_PTR __ilshift__)();
LANTERN_API void (LANTERN_PTR __ilshift__)();
LANTERN_API void (LANTERN_PTR __rshift__)();
LANTERN_API void (LANTERN_PTR __rshift__)();
LANTERN_API void (LANTERN_PTR __irshift__)();
LANTERN_API void (LANTERN_PTR __irshift__)();
LANTERN_API void (LANTERN_PTR lgamma_)();
LANTERN_API void (LANTERN_PTR atan2_)();
LANTERN_API void (LANTERN_PTR tril_)();
LANTERN_API void (LANTERN_PTR triu_)();
LANTERN_API void (LANTERN_PTR digamma_)();
LANTERN_API void (LANTERN_PTR polygamma_)();
LANTERN_API void (LANTERN_PTR renorm_)();
LANTERN_API void (LANTERN_PTR pow_)();
LANTERN_API void (LANTERN_PTR pow_)();
LANTERN_API void (LANTERN_PTR lerp_)();
LANTERN_API void (LANTERN_PTR lerp_)();
LANTERN_API void (LANTERN_PTR fmod_)();
LANTERN_API void (LANTERN_PTR fmod_)();
LANTERN_API void (LANTERN_PTR remainder_)();
LANTERN_API void (LANTERN_PTR remainder_)();
LANTERN_API void (LANTERN_PTR addbmm_)();
LANTERN_API void (LANTERN_PTR addbmm_out)();
LANTERN_API void (LANTERN_PTR addbmm)();
LANTERN_API void (LANTERN_PTR addcdiv_)();
LANTERN_API void (LANTERN_PTR random_)();
LANTERN_API void (LANTERN_PTR random_)();
LANTERN_API void (LANTERN_PTR random_)();
LANTERN_API void (LANTERN_PTR uniform_)();
LANTERN_API void (LANTERN_PTR normal_)();
LANTERN_API void (LANTERN_PTR cauchy_)();
LANTERN_API void (LANTERN_PTR log_normal_)();
LANTERN_API void (LANTERN_PTR exponential_)();
LANTERN_API void (LANTERN_PTR geometric_)();
LANTERN_API void (LANTERN_PTR diag_out)();
LANTERN_API void (LANTERN_PTR diag)();
LANTERN_API void (LANTERN_PTR cross_out)();
LANTERN_API void (LANTERN_PTR cross)();
LANTERN_API void (LANTERN_PTR triu_out)();
LANTERN_API void (LANTERN_PTR triu)();
LANTERN_API void (LANTERN_PTR tril_out)();
LANTERN_API void (LANTERN_PTR tril)();
LANTERN_API void (LANTERN_PTR tril_indices)();
LANTERN_API void (LANTERN_PTR triu_indices)();
LANTERN_API void (LANTERN_PTR trace)();
LANTERN_API void (LANTERN_PTR ne_out)();
LANTERN_API void (LANTERN_PTR ne)();
LANTERN_API void (LANTERN_PTR ne_out)();
LANTERN_API void (LANTERN_PTR ne)();
LANTERN_API void (LANTERN_PTR eq_out)();
LANTERN_API void (LANTERN_PTR eq)();
LANTERN_API void (LANTERN_PTR eq_out)();
LANTERN_API void (LANTERN_PTR eq)();
LANTERN_API void (LANTERN_PTR ge_out)();
LANTERN_API void (LANTERN_PTR ge)();
LANTERN_API void (LANTERN_PTR ge_out)();
LANTERN_API void (LANTERN_PTR ge)();
LANTERN_API void (LANTERN_PTR le_out)();
LANTERN_API void (LANTERN_PTR le)();
LANTERN_API void (LANTERN_PTR le_out)();
LANTERN_API void (LANTERN_PTR le)();
LANTERN_API void (LANTERN_PTR gt_out)();
LANTERN_API void (LANTERN_PTR gt)();
LANTERN_API void (LANTERN_PTR gt_out)();
LANTERN_API void (LANTERN_PTR gt)();
LANTERN_API void (LANTERN_PTR lt_out)();
LANTERN_API void (LANTERN_PTR lt)();
LANTERN_API void (LANTERN_PTR lt_out)();
LANTERN_API void (LANTERN_PTR lt)();
LANTERN_API void (LANTERN_PTR take_out)();
LANTERN_API void (LANTERN_PTR take)();
LANTERN_API void (LANTERN_PTR index_select_out)();
LANTERN_API void (LANTERN_PTR index_select)();
LANTERN_API void (LANTERN_PTR index_select_out)();
LANTERN_API void (LANTERN_PTR index_select)();
LANTERN_API void (LANTERN_PTR masked_select_out)();
LANTERN_API void (LANTERN_PTR masked_select)();
LANTERN_API void (LANTERN_PTR nonzero_out)();
LANTERN_API void (LANTERN_PTR nonzero)();
LANTERN_API void (LANTERN_PTR nonzero_numpy)();
LANTERN_API void (LANTERN_PTR gather_out)();
LANTERN_API void (LANTERN_PTR gather)();
LANTERN_API void (LANTERN_PTR gather_out)();
LANTERN_API void (LANTERN_PTR gather)();
LANTERN_API void (LANTERN_PTR _gather_sparse_backward)();
LANTERN_API void (LANTERN_PTR addcmul_out)();
LANTERN_API void (LANTERN_PTR addcmul)();
LANTERN_API void (LANTERN_PTR addcmul_)();
LANTERN_API void (LANTERN_PTR addcdiv_out)();
LANTERN_API void (LANTERN_PTR addcdiv)();
LANTERN_API void (LANTERN_PTR lstsq_out)();
LANTERN_API void (LANTERN_PTR lstsq)();
LANTERN_API void (LANTERN_PTR triangular_solve_out)();
LANTERN_API void (LANTERN_PTR triangular_solve)();
LANTERN_API void (LANTERN_PTR _triangular_solve_helper)();
LANTERN_API void (LANTERN_PTR symeig_out)();
LANTERN_API void (LANTERN_PTR symeig)();
LANTERN_API void (LANTERN_PTR _symeig_helper)();
LANTERN_API void (LANTERN_PTR eig_out)();
LANTERN_API void (LANTERN_PTR eig)();
LANTERN_API void (LANTERN_PTR svd_out)();
LANTERN_API void (LANTERN_PTR svd)();
LANTERN_API void (LANTERN_PTR _svd_helper)();
LANTERN_API void (LANTERN_PTR cholesky_out)();
LANTERN_API void (LANTERN_PTR cholesky)();
LANTERN_API void (LANTERN_PTR _cholesky_helper)();
LANTERN_API void (LANTERN_PTR cholesky_solve_out)();
LANTERN_API void (LANTERN_PTR cholesky_solve)();
LANTERN_API void (LANTERN_PTR _cholesky_solve_helper)();
LANTERN_API void (LANTERN_PTR solve)();
LANTERN_API void (LANTERN_PTR solve_out)();
LANTERN_API void (LANTERN_PTR _solve_helper)();
LANTERN_API void (LANTERN_PTR cholesky_inverse_out)();
LANTERN_API void (LANTERN_PTR cholesky_inverse)();
LANTERN_API void (LANTERN_PTR qr_out)();
LANTERN_API void (LANTERN_PTR qr)();
LANTERN_API void (LANTERN_PTR _qr_helper)();
LANTERN_API void (LANTERN_PTR geqrf_out)();
LANTERN_API void (LANTERN_PTR geqrf)();
LANTERN_API void (LANTERN_PTR orgqr_out)();
LANTERN_API void (LANTERN_PTR orgqr)();
LANTERN_API void (LANTERN_PTR ormqr_out)();
LANTERN_API void (LANTERN_PTR ormqr)();
LANTERN_API void (LANTERN_PTR _lu_with_info)();
LANTERN_API void (LANTERN_PTR lu_solve_out)();
LANTERN_API void (LANTERN_PTR lu_solve)();
LANTERN_API void (LANTERN_PTR _lu_solve_helper)();
LANTERN_API void (LANTERN_PTR multinomial_out)();
LANTERN_API void (LANTERN_PTR multinomial)();
LANTERN_API void (LANTERN_PTR _multinomial_alias_setup)();
LANTERN_API void (LANTERN_PTR _multinomial_alias_draw)();
LANTERN_API void (LANTERN_PTR lgamma_out)();
LANTERN_API void (LANTERN_PTR lgamma)();
LANTERN_API void (LANTERN_PTR digamma_out)();
LANTERN_API void (LANTERN_PTR digamma)();
LANTERN_API void (LANTERN_PTR polygamma_out)();
LANTERN_API void (LANTERN_PTR polygamma)();
LANTERN_API void (LANTERN_PTR erfinv)();
LANTERN_API void (LANTERN_PTR erfinv_)();
LANTERN_API void (LANTERN_PTR erfinv_out)();
LANTERN_API void (LANTERN_PTR sign)();
LANTERN_API void (LANTERN_PTR sign_)();
LANTERN_API void (LANTERN_PTR sign_out)();
LANTERN_API void (LANTERN_PTR dist)();
LANTERN_API void (LANTERN_PTR atan2_out)();
LANTERN_API void (LANTERN_PTR atan2)();
LANTERN_API void (LANTERN_PTR lerp_out)();
LANTERN_API void (LANTERN_PTR lerp_out)();
LANTERN_API void (LANTERN_PTR lerp)();
LANTERN_API void (LANTERN_PTR lerp)();
LANTERN_API void (LANTERN_PTR histc_out)();
LANTERN_API void (LANTERN_PTR histc)();
LANTERN_API void (LANTERN_PTR fmod_out)();
LANTERN_API void (LANTERN_PTR fmod)();
LANTERN_API void (LANTERN_PTR fmod_out)();
LANTERN_API void (LANTERN_PTR fmod)();
LANTERN_API void (LANTERN_PTR remainder_out)();
LANTERN_API void (LANTERN_PTR remainder)();
LANTERN_API void (LANTERN_PTR remainder_out)();
LANTERN_API void (LANTERN_PTR remainder)();
LANTERN_API void (LANTERN_PTR min_out)();
LANTERN_API void (LANTERN_PTR min)();
LANTERN_API void (LANTERN_PTR min)();
LANTERN_API void (LANTERN_PTR max_out)();
LANTERN_API void (LANTERN_PTR max)();
LANTERN_API void (LANTERN_PTR max)();
LANTERN_API void (LANTERN_PTR median)();
LANTERN_API void (LANTERN_PTR sort_out)();
LANTERN_API void (LANTERN_PTR sort)();
LANTERN_API void (LANTERN_PTR sort_out)();
LANTERN_API void (LANTERN_PTR sort)();
LANTERN_API void (LANTERN_PTR argsort)();
LANTERN_API void (LANTERN_PTR argsort)();
LANTERN_API void (LANTERN_PTR topk_out)();
LANTERN_API void (LANTERN_PTR topk)();
LANTERN_API void (LANTERN_PTR all)();
LANTERN_API void (LANTERN_PTR any)();
LANTERN_API void (LANTERN_PTR renorm_out)();
LANTERN_API void (LANTERN_PTR renorm)();
LANTERN_API void (LANTERN_PTR unfold)();
LANTERN_API void (LANTERN_PTR equal)();
LANTERN_API void (LANTERN_PTR pow_out)();
LANTERN_API void (LANTERN_PTR pow)();
LANTERN_API void (LANTERN_PTR pow_out)();
LANTERN_API void (LANTERN_PTR pow)();
LANTERN_API void (LANTERN_PTR normal_out)();
LANTERN_API void (LANTERN_PTR normal)();
LANTERN_API void (LANTERN_PTR normal_out)();
LANTERN_API void (LANTERN_PTR normal)();
LANTERN_API void (LANTERN_PTR normal_out)();
LANTERN_API void (LANTERN_PTR normal)();
LANTERN_API void (LANTERN_PTR normal)();
LANTERN_API void (LANTERN_PTR normal_out)();
LANTERN_API void (LANTERN_PTR alias)();
LANTERN_API void (LANTERN_PTR _addr)();
LANTERN_API void (LANTERN_PTR _addr_)();
LANTERN_API void (LANTERN_PTR _addr_out)();
LANTERN_API void (LANTERN_PTR _index_copy_)();
LANTERN_API void (LANTERN_PTR _cumsum)();
LANTERN_API void (LANTERN_PTR _cumsum_out)();
LANTERN_API void (LANTERN_PTR _cumprod)();
LANTERN_API void (LANTERN_PTR _cumprod_out)();
LANTERN_API void (LANTERN_PTR _var)();
LANTERN_API void (LANTERN_PTR _std)();
LANTERN_API void (LANTERN_PTR _cat)();
LANTERN_API void (LANTERN_PTR _cat_out)();
LANTERN_API void (LANTERN_PTR _mode)();
LANTERN_API void (LANTERN_PTR _mode_out)();
LANTERN_API void (LANTERN_PTR _max)();
LANTERN_API void (LANTERN_PTR _max_out)();
LANTERN_API void (LANTERN_PTR _min)();
LANTERN_API void (LANTERN_PTR _min_out)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy_out)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy_backward_out)();
LANTERN_API void (LANTERN_PTR binary_cross_entropy_backward)();
LANTERN_API void (LANTERN_PTR mse_loss_out)();
LANTERN_API void (LANTERN_PTR mse_loss)();
LANTERN_API void (LANTERN_PTR mse_loss_backward_out)();
LANTERN_API void (LANTERN_PTR mse_loss_backward)();
LANTERN_API void (LANTERN_PTR l1_loss_out)();
LANTERN_API void (LANTERN_PTR l1_loss)();
LANTERN_API void (LANTERN_PTR l1_loss_backward_out)();
LANTERN_API void (LANTERN_PTR l1_loss_backward)();
LANTERN_API void (LANTERN_PTR multi_margin_loss_out)();
LANTERN_API void (LANTERN_PTR multi_margin_loss)();
LANTERN_API void (LANTERN_PTR multi_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR multi_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss_out)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss_forward_out)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss_forward)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR multilabel_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR nll_loss_out)();
LANTERN_API void (LANTERN_PTR nll_loss)();
LANTERN_API void (LANTERN_PTR nll_loss_forward_out)();
LANTERN_API void (LANTERN_PTR nll_loss_forward)();
LANTERN_API void (LANTERN_PTR nll_loss_backward_out)();
LANTERN_API void (LANTERN_PTR nll_loss_backward)();
LANTERN_API void (LANTERN_PTR nll_loss2d_out)();
LANTERN_API void (LANTERN_PTR nll_loss2d)();
LANTERN_API void (LANTERN_PTR nll_loss2d_forward_out)();
LANTERN_API void (LANTERN_PTR nll_loss2d_forward)();
LANTERN_API void (LANTERN_PTR nll_loss2d_backward_out)();
LANTERN_API void (LANTERN_PTR nll_loss2d_backward)();
LANTERN_API void (LANTERN_PTR smooth_l1_loss_out)();
LANTERN_API void (LANTERN_PTR smooth_l1_loss)();
LANTERN_API void (LANTERN_PTR smooth_l1_loss_backward_out)();
LANTERN_API void (LANTERN_PTR smooth_l1_loss_backward)();
LANTERN_API void (LANTERN_PTR soft_margin_loss_out)();
LANTERN_API void (LANTERN_PTR soft_margin_loss)();
LANTERN_API void (LANTERN_PTR soft_margin_loss_backward_out)();
LANTERN_API void (LANTERN_PTR soft_margin_loss_backward)();
LANTERN_API void (LANTERN_PTR elu_out)();
LANTERN_API void (LANTERN_PTR elu)();
LANTERN_API void (LANTERN_PTR elu_backward_out)();
LANTERN_API void (LANTERN_PTR elu_backward)();
LANTERN_API void (LANTERN_PTR elu_)();
LANTERN_API void (LANTERN_PTR glu_out)();
LANTERN_API void (LANTERN_PTR glu)();
LANTERN_API void (LANTERN_PTR glu_backward_out)();
LANTERN_API void (LANTERN_PTR glu_backward)();
LANTERN_API void (LANTERN_PTR hardtanh_out)();
LANTERN_API void (LANTERN_PTR hardtanh)();
LANTERN_API void (LANTERN_PTR hardtanh_backward_out)();
LANTERN_API void (LANTERN_PTR hardtanh_backward)();
LANTERN_API void (LANTERN_PTR hardtanh_)();
LANTERN_API void (LANTERN_PTR leaky_relu_out)();
LANTERN_API void (LANTERN_PTR leaky_relu)();
LANTERN_API void (LANTERN_PTR leaky_relu_backward_out)();
LANTERN_API void (LANTERN_PTR leaky_relu_backward)();
LANTERN_API void (LANTERN_PTR leaky_relu_)();
LANTERN_API void (LANTERN_PTR log_sigmoid_out)();
LANTERN_API void (LANTERN_PTR log_sigmoid)();
LANTERN_API void (LANTERN_PTR log_sigmoid_forward_out)();
LANTERN_API void (LANTERN_PTR log_sigmoid_forward)();
LANTERN_API void (LANTERN_PTR log_sigmoid_backward_out)();
LANTERN_API void (LANTERN_PTR log_sigmoid_backward)();
LANTERN_API void (LANTERN_PTR rrelu_with_noise_out)();
LANTERN_API void (LANTERN_PTR rrelu_with_noise)();
LANTERN_API void (LANTERN_PTR rrelu_with_noise_backward_out)();
LANTERN_API void (LANTERN_PTR rrelu_with_noise_backward)();
LANTERN_API void (LANTERN_PTR rrelu_with_noise_)();
LANTERN_API void (LANTERN_PTR softplus_out)();
LANTERN_API void (LANTERN_PTR softplus)();
LANTERN_API void (LANTERN_PTR softplus_backward_out)();
LANTERN_API void (LANTERN_PTR softplus_backward)();
LANTERN_API void (LANTERN_PTR softshrink_out)();
LANTERN_API void (LANTERN_PTR softshrink)();
LANTERN_API void (LANTERN_PTR softshrink_backward_out)();
LANTERN_API void (LANTERN_PTR softshrink_backward)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool2d_out)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR mkldnn_adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR _adaptive_avg_pool2d)();
LANTERN_API void (LANTERN_PTR _adaptive_avg_pool2d_backward)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool3d_out)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool3d)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR adaptive_avg_pool3d_backward)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool2d_out)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool2d)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool2d_backward)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool3d_out)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool3d)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR adaptive_max_pool3d_backward)();
LANTERN_API void (LANTERN_PTR avg_pool2d_out)();
LANTERN_API void (LANTERN_PTR avg_pool2d)();
LANTERN_API void (LANTERN_PTR avg_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR avg_pool2d_backward)();
LANTERN_API void (LANTERN_PTR avg_pool3d_out)();
LANTERN_API void (LANTERN_PTR avg_pool3d)();
LANTERN_API void (LANTERN_PTR avg_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR avg_pool3d_backward)();
LANTERN_API void (LANTERN_PTR fractional_max_pool2d_out)();
LANTERN_API void (LANTERN_PTR fractional_max_pool2d)();
LANTERN_API void (LANTERN_PTR fractional_max_pool2d_backward_out)();
LANTERN_API void (LANTERN_PTR fractional_max_pool2d_backward)();
LANTERN_API void (LANTERN_PTR fractional_max_pool3d_out)();
LANTERN_API void (LANTERN_PTR fractional_max_pool3d)();
LANTERN_API void (LANTERN_PTR fractional_max_pool3d_backward_out)();
LANTERN_API void (LANTERN_PTR fractional_max_pool3d_backward)();
LANTERN_API void (LANTERN_PTR max_pool2d_with_indices_out)();
LANTERN_API void (LANTERN_PTR max_pool2d_with_indices)();
LANTERN_API void (LANTERN_PTR max_pool2d_with_indices_backward_out)();
LANTERN_API void (LANTERN_PTR max_pool2d_with_indices_backward)();
LANTERN_API void (LANTERN_PTR max_pool3d_with_indices_out)();
LANTERN_API void (LANTERN_PTR max_pool3d_with_indices)();
LANTERN_API void (LANTERN_PTR max_pool3d_with_indices_backward_out)();
LANTERN_API void (LANTERN_PTR max_pool3d_with_indices_backward)();
LANTERN_API void (LANTERN_PTR max_unpool2d_out)();
LANTERN_API void (LANTERN_PTR max_unpool2d)();
LANTERN_API void (LANTERN_PTR max_unpool2d_backward_out)();
LANTERN_API void (LANTERN_PTR max_unpool2d_backward)();
LANTERN_API void (LANTERN_PTR max_unpool3d_out)();
LANTERN_API void (LANTERN_PTR max_unpool3d)();
LANTERN_API void (LANTERN_PTR max_unpool3d_backward_out)();
LANTERN_API void (LANTERN_PTR max_unpool3d_backward)();
LANTERN_API void (LANTERN_PTR reflection_pad1d_out)();
LANTERN_API void (LANTERN_PTR reflection_pad1d)();
LANTERN_API void (LANTERN_PTR reflection_pad1d_backward_out)();
LANTERN_API void (LANTERN_PTR reflection_pad1d_backward)();
LANTERN_API void (LANTERN_PTR reflection_pad2d_out)();
LANTERN_API void (LANTERN_PTR reflection_pad2d)();
LANTERN_API void (LANTERN_PTR reflection_pad2d_backward_out)();
LANTERN_API void (LANTERN_PTR reflection_pad2d_backward)();
LANTERN_API void (LANTERN_PTR replication_pad1d_out)();
LANTERN_API void (LANTERN_PTR replication_pad1d)();
LANTERN_API void (LANTERN_PTR replication_pad1d_backward_out)();
LANTERN_API void (LANTERN_PTR replication_pad1d_backward)();
LANTERN_API void (LANTERN_PTR replication_pad2d_out)();
LANTERN_API void (LANTERN_PTR replication_pad2d)();
LANTERN_API void (LANTERN_PTR replication_pad2d_backward_out)();
LANTERN_API void (LANTERN_PTR replication_pad2d_backward)();
LANTERN_API void (LANTERN_PTR replication_pad3d_out)();
LANTERN_API void (LANTERN_PTR replication_pad3d)();
LANTERN_API void (LANTERN_PTR replication_pad3d_backward_out)();
LANTERN_API void (LANTERN_PTR replication_pad3d_backward)();
LANTERN_API void (LANTERN_PTR upsample_linear1d_out)();
LANTERN_API void (LANTERN_PTR upsample_linear1d)();
LANTERN_API void (LANTERN_PTR upsample_linear1d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_linear1d_backward)();
LANTERN_API void (LANTERN_PTR upsample_bilinear2d_out)();
LANTERN_API void (LANTERN_PTR upsample_bilinear2d)();
LANTERN_API void (LANTERN_PTR upsample_bilinear2d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_bilinear2d_backward)();
LANTERN_API void (LANTERN_PTR upsample_bicubic2d_out)();
LANTERN_API void (LANTERN_PTR upsample_bicubic2d)();
LANTERN_API void (LANTERN_PTR upsample_bicubic2d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_bicubic2d_backward)();
LANTERN_API void (LANTERN_PTR upsample_trilinear3d_out)();
LANTERN_API void (LANTERN_PTR upsample_trilinear3d)();
LANTERN_API void (LANTERN_PTR upsample_trilinear3d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_trilinear3d_backward)();
LANTERN_API void (LANTERN_PTR upsample_nearest1d_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest1d)();
LANTERN_API void (LANTERN_PTR upsample_nearest1d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest1d_backward)();
LANTERN_API void (LANTERN_PTR upsample_nearest2d_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest2d)();
LANTERN_API void (LANTERN_PTR upsample_nearest2d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest2d_backward)();
LANTERN_API void (LANTERN_PTR upsample_nearest3d_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest3d)();
LANTERN_API void (LANTERN_PTR upsample_nearest3d_backward_out)();
LANTERN_API void (LANTERN_PTR upsample_nearest3d_backward)();
LANTERN_API void (LANTERN_PTR sigmoid_backward_out)();
LANTERN_API void (LANTERN_PTR sigmoid_backward)();
LANTERN_API void (LANTERN_PTR tanh_backward_out)();
LANTERN_API void (LANTERN_PTR tanh_backward)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose2d_out)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose2d)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose2d_backward_out)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose2d_backward)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose3d_out)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose3d)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose3d_backward_out)();
LANTERN_API void (LANTERN_PTR slow_conv_transpose3d_backward)();
LANTERN_API void (LANTERN_PTR thnn_conv2d_out)();
LANTERN_API void (LANTERN_PTR thnn_conv2d)();
LANTERN_API void (LANTERN_PTR thnn_conv2d_forward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv2d_forward)();
LANTERN_API void (LANTERN_PTR thnn_conv2d_backward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv2d_backward)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d_out)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d_forward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d_forward)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d_backward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv_depthwise2d_backward)();
LANTERN_API void (LANTERN_PTR thnn_conv3d_out)();
LANTERN_API void (LANTERN_PTR thnn_conv3d)();
LANTERN_API void (LANTERN_PTR thnn_conv3d_forward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv3d_forward)();
LANTERN_API void (LANTERN_PTR thnn_conv3d_backward_out)();
LANTERN_API void (LANTERN_PTR thnn_conv3d_backward)();
LANTERN_API void (LANTERN_PTR slow_conv_dilated2d)();
LANTERN_API void (LANTERN_PTR slow_conv_dilated2d_backward)();
LANTERN_API void (LANTERN_PTR slow_conv_dilated3d)();
LANTERN_API void (LANTERN_PTR slow_conv_dilated3d_backward)();
LANTERN_API void (LANTERN_PTR col2im_out)();
LANTERN_API void (LANTERN_PTR col2im)();
LANTERN_API void (LANTERN_PTR col2im_backward_out)();
LANTERN_API void (LANTERN_PTR col2im_backward)();
LANTERN_API void (LANTERN_PTR im2col_out)();
LANTERN_API void (LANTERN_PTR im2col)();
LANTERN_API void (LANTERN_PTR im2col_backward_out)();
LANTERN_API void (LANTERN_PTR im2col_backward)();
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
  LOAD_SYMBOL(_cast_Byte)
  LOAD_SYMBOL(_cast_Char)
  LOAD_SYMBOL(_cast_Double)
  LOAD_SYMBOL(_cast_Float)
  LOAD_SYMBOL(_cast_Int)
  LOAD_SYMBOL(_cast_Long)
  LOAD_SYMBOL(_cast_Short)
  LOAD_SYMBOL(_cast_Half)
  LOAD_SYMBOL(backward)
  LOAD_SYMBOL(set_data)
  LOAD_SYMBOL(data)
  LOAD_SYMBOL(is_leaf)
  LOAD_SYMBOL(output_nr)
  LOAD_SYMBOL(_version)
  LOAD_SYMBOL(rename_)
  LOAD_SYMBOL(rename)
  LOAD_SYMBOL(align_to)
  LOAD_SYMBOL(align_as)
  LOAD_SYMBOL(align_tensors)
  LOAD_SYMBOL(refine_names)
  LOAD_SYMBOL(unflatten)
  LOAD_SYMBOL(unflatten)
  LOAD_SYMBOL(_cudnn_ctc_loss)
  LOAD_SYMBOL(_cudnn_rnn_flatten_weight)
  LOAD_SYMBOL(_cudnn_rnn)
  LOAD_SYMBOL(_cudnn_rnn_backward)
  LOAD_SYMBOL(_cudnn_init_dropout_state)
  LOAD_SYMBOL(_debug_has_internal_overlap)
  LOAD_SYMBOL(_fused_dropout)
  LOAD_SYMBOL(_masked_scale)
  LOAD_SYMBOL(_sobol_engine_draw)
  LOAD_SYMBOL(_sobol_engine_ff_)
  LOAD_SYMBOL(_sobol_engine_scramble_)
  LOAD_SYMBOL(_sobol_engine_initialize_state_)
  LOAD_SYMBOL(_reshape_from_tensor)
  LOAD_SYMBOL(_shape_as_tensor)
  LOAD_SYMBOL(dropout)
  LOAD_SYMBOL(dropout_)
  LOAD_SYMBOL(feature_dropout)
  LOAD_SYMBOL(feature_dropout_)
  LOAD_SYMBOL(alpha_dropout)
  LOAD_SYMBOL(alpha_dropout_)
  LOAD_SYMBOL(feature_alpha_dropout)
  LOAD_SYMBOL(feature_alpha_dropout_)
  LOAD_SYMBOL(abs)
  LOAD_SYMBOL(abs_)
  LOAD_SYMBOL(abs_out)
  LOAD_SYMBOL(acos)
  LOAD_SYMBOL(acos_)
  LOAD_SYMBOL(acos_out)
  LOAD_SYMBOL(avg_pool1d)
  LOAD_SYMBOL(adaptive_avg_pool1d)
  LOAD_SYMBOL(adaptive_max_pool1d)
  LOAD_SYMBOL(add)
  LOAD_SYMBOL(add_)
  LOAD_SYMBOL(add_out)
  LOAD_SYMBOL(add)
  LOAD_SYMBOL(add_)
  LOAD_SYMBOL(addmv)
  LOAD_SYMBOL(addmv_)
  LOAD_SYMBOL(addmv_out)
  LOAD_SYMBOL(addr)
  LOAD_SYMBOL(addr_)
  LOAD_SYMBOL(addr_out)
  LOAD_SYMBOL(affine_grid_generator)
  LOAD_SYMBOL(affine_grid_generator_backward)
  LOAD_SYMBOL(all)
  LOAD_SYMBOL(all_out)
  LOAD_SYMBOL(all)
  LOAD_SYMBOL(all_out)
  LOAD_SYMBOL(allclose)
  LOAD_SYMBOL(any)
  LOAD_SYMBOL(any_out)
  LOAD_SYMBOL(any)
  LOAD_SYMBOL(any_out)
  LOAD_SYMBOL(arange)
  LOAD_SYMBOL(arange)
  LOAD_SYMBOL(arange)
  LOAD_SYMBOL(arange_out)
  LOAD_SYMBOL(arange_out)
  LOAD_SYMBOL(_dim_arange)
  LOAD_SYMBOL(argmax)
  LOAD_SYMBOL(argmin)
  LOAD_SYMBOL(as_strided)
  LOAD_SYMBOL(as_strided_)
  LOAD_SYMBOL(asin)
  LOAD_SYMBOL(asin_)
  LOAD_SYMBOL(asin_out)
  LOAD_SYMBOL(atan)
  LOAD_SYMBOL(atan_)
  LOAD_SYMBOL(atan_out)
  LOAD_SYMBOL(baddbmm)
  LOAD_SYMBOL(baddbmm_)
  LOAD_SYMBOL(_baddbmm_mkl_)
  LOAD_SYMBOL(baddbmm_out)
  LOAD_SYMBOL(bartlett_window)
  LOAD_SYMBOL(bartlett_window)
  LOAD_SYMBOL(batch_norm)
  LOAD_SYMBOL(_batch_norm_impl_index)
  LOAD_SYMBOL(_batch_norm_impl_index_backward)
  LOAD_SYMBOL(bernoulli)
  LOAD_SYMBOL(bernoulli_out)
  LOAD_SYMBOL(bernoulli_)
  LOAD_SYMBOL(bernoulli_)
  LOAD_SYMBOL(bernoulli)
  LOAD_SYMBOL(bilinear)
  LOAD_SYMBOL(binary_cross_entropy_with_logits)
  LOAD_SYMBOL(binary_cross_entropy_with_logits_backward)
  LOAD_SYMBOL(bincount)
  LOAD_SYMBOL(bitwise_not)
  LOAD_SYMBOL(bitwise_not_)
  LOAD_SYMBOL(bitwise_not_out)
  LOAD_SYMBOL(logical_not)
  LOAD_SYMBOL(logical_not_)
  LOAD_SYMBOL(logical_not_out)
  LOAD_SYMBOL(logical_xor)
  LOAD_SYMBOL(logical_xor_)
  LOAD_SYMBOL(logical_xor_out)
  LOAD_SYMBOL(blackman_window)
  LOAD_SYMBOL(blackman_window)
  LOAD_SYMBOL(bmm)
  LOAD_SYMBOL(bmm_out)
  LOAD_SYMBOL(broadcast_tensors)
  LOAD_SYMBOL(cat)
  LOAD_SYMBOL(cat_out)
  LOAD_SYMBOL(cat)
  LOAD_SYMBOL(cat_out)
  LOAD_SYMBOL(ceil)
  LOAD_SYMBOL(ceil_)
  LOAD_SYMBOL(ceil_out)
  LOAD_SYMBOL(chain_matmul)
  LOAD_SYMBOL(chunk)
  LOAD_SYMBOL(clamp)
  LOAD_SYMBOL(clamp_)
  LOAD_SYMBOL(clamp_out)
  LOAD_SYMBOL(clamp_max)
  LOAD_SYMBOL(clamp_max_)
  LOAD_SYMBOL(clamp_max_out)
  LOAD_SYMBOL(clamp_min)
  LOAD_SYMBOL(clamp_min_)
  LOAD_SYMBOL(clamp_min_out)
  LOAD_SYMBOL(cudnn_is_acceptable)
  LOAD_SYMBOL(constant_pad_nd)
  LOAD_SYMBOL(contiguous)
  LOAD_SYMBOL(convolution)
  LOAD_SYMBOL(convolution_overrideable)
  LOAD_SYMBOL(convolution_backward_overrideable)
  LOAD_SYMBOL(_convolution)
  LOAD_SYMBOL(_convolution_nogroup)
  LOAD_SYMBOL(_convolution_double_backward)
  LOAD_SYMBOL(conv1d)
  LOAD_SYMBOL(conv2d)
  LOAD_SYMBOL(conv3d)
  LOAD_SYMBOL(conv_tbc)
  LOAD_SYMBOL(conv_tbc_backward)
  LOAD_SYMBOL(conv_transpose1d)
  LOAD_SYMBOL(conv_transpose2d)
  LOAD_SYMBOL(conv_transpose3d)
  LOAD_SYMBOL(copy_)
  LOAD_SYMBOL(_copy_from)
  LOAD_SYMBOL(cos)
  LOAD_SYMBOL(cos_)
  LOAD_SYMBOL(cos_out)
  LOAD_SYMBOL(cosh)
  LOAD_SYMBOL(cosh_)
  LOAD_SYMBOL(cosh_out)
  LOAD_SYMBOL(cosine_embedding_loss)
  LOAD_SYMBOL(cudnn_affine_grid_generator)
  LOAD_SYMBOL(cudnn_affine_grid_generator_backward)
  LOAD_SYMBOL(cudnn_batch_norm)
  LOAD_SYMBOL(cudnn_batch_norm_backward)
  LOAD_SYMBOL(cudnn_convolution)
  LOAD_SYMBOL(cudnn_convolution_backward_input)
  LOAD_SYMBOL(cudnn_convolution_backward)
  LOAD_SYMBOL(cudnn_convolution_backward_bias)
  LOAD_SYMBOL(cudnn_convolution_backward_weight)
  LOAD_SYMBOL(cudnn_convolution_transpose)
  LOAD_SYMBOL(cudnn_convolution_transpose_backward)
  LOAD_SYMBOL(cudnn_convolution_transpose_backward_bias)
  LOAD_SYMBOL(cudnn_convolution_transpose_backward_input)
  LOAD_SYMBOL(cudnn_convolution_transpose_backward_weight)
  LOAD_SYMBOL(cudnn_grid_sampler)
  LOAD_SYMBOL(cudnn_grid_sampler_backward)
  LOAD_SYMBOL(cumsum)
  LOAD_SYMBOL(cumsum_out)
  LOAD_SYMBOL(cumsum)
  LOAD_SYMBOL(cumsum_out)
  LOAD_SYMBOL(cumprod)
  LOAD_SYMBOL(cumprod_out)
  LOAD_SYMBOL(cumprod)
  LOAD_SYMBOL(cumprod_out)
  LOAD_SYMBOL(ctc_loss)
  LOAD_SYMBOL(ctc_loss)
  LOAD_SYMBOL(_ctc_loss)
  LOAD_SYMBOL(_ctc_loss_backward)
  LOAD_SYMBOL(det)
  LOAD_SYMBOL(diag_embed)
  LOAD_SYMBOL(diagflat)
  LOAD_SYMBOL(diagonal)
  LOAD_SYMBOL(fill_diagonal_)
  LOAD_SYMBOL(div)
  LOAD_SYMBOL(div_)
  LOAD_SYMBOL(div_out)
  LOAD_SYMBOL(div)
  LOAD_SYMBOL(div_)
  LOAD_SYMBOL(dot)
  LOAD_SYMBOL(dot_out)
  LOAD_SYMBOL(einsum)
  LOAD_SYMBOL(embedding)
  LOAD_SYMBOL(embedding_backward)
  LOAD_SYMBOL(embedding_dense_backward)
  LOAD_SYMBOL(embedding_renorm_)
  LOAD_SYMBOL(embedding_sparse_backward)
  LOAD_SYMBOL(embedding_bag)
  LOAD_SYMBOL(_embedding_bag)
  LOAD_SYMBOL(_embedding_bag_backward)
  LOAD_SYMBOL(_embedding_bag_sparse_backward)
  LOAD_SYMBOL(_embedding_bag_dense_backward)
  LOAD_SYMBOL(_embedding_bag_per_sample_weights_backward)
  LOAD_SYMBOL(empty)
  LOAD_SYMBOL(empty)
  LOAD_SYMBOL(new_empty)
  LOAD_SYMBOL(new_full)
  LOAD_SYMBOL(_empty_affine_quantized)
  LOAD_SYMBOL(_empty_per_channel_affine_quantized)
  LOAD_SYMBOL(resize_)
  LOAD_SYMBOL(empty_out)
  LOAD_SYMBOL(empty_like)
  LOAD_SYMBOL(empty_like)
  LOAD_SYMBOL(empty_strided)
  LOAD_SYMBOL(erf)
  LOAD_SYMBOL(erf_)
  LOAD_SYMBOL(erf_out)
  LOAD_SYMBOL(erfc)
  LOAD_SYMBOL(erfc_)
  LOAD_SYMBOL(erfc_out)
  LOAD_SYMBOL(exp)
  LOAD_SYMBOL(exp_)
  LOAD_SYMBOL(exp_out)
  LOAD_SYMBOL(expm1)
  LOAD_SYMBOL(expm1_)
  LOAD_SYMBOL(expm1_out)
  LOAD_SYMBOL(expand)
  LOAD_SYMBOL(expand_as)
  LOAD_SYMBOL(eye)
  LOAD_SYMBOL(eye)
  LOAD_SYMBOL(eye_out)
  LOAD_SYMBOL(eye_out)
  LOAD_SYMBOL(flatten)
  LOAD_SYMBOL(flatten)
  LOAD_SYMBOL(flatten)
  LOAD_SYMBOL(flatten)
  LOAD_SYMBOL(fill_)
  LOAD_SYMBOL(fill_)
  LOAD_SYMBOL(floor)
  LOAD_SYMBOL(floor_)
  LOAD_SYMBOL(floor_out)
  LOAD_SYMBOL(frac)
  LOAD_SYMBOL(frac_)
  LOAD_SYMBOL(frac_out)
  LOAD_SYMBOL(full)
  LOAD_SYMBOL(full)
  LOAD_SYMBOL(full_out)
  LOAD_SYMBOL(full_like)
  LOAD_SYMBOL(full_like)
  LOAD_SYMBOL(from_file)
  LOAD_SYMBOL(grid_sampler)
  LOAD_SYMBOL(grid_sampler_2d)
  LOAD_SYMBOL(grid_sampler_2d_backward)
  LOAD_SYMBOL(grid_sampler_3d)
  LOAD_SYMBOL(grid_sampler_3d_backward)
  LOAD_SYMBOL(hann_window)
  LOAD_SYMBOL(hann_window)
  LOAD_SYMBOL(hamming_window)
  LOAD_SYMBOL(hamming_window)
  LOAD_SYMBOL(hamming_window)
  LOAD_SYMBOL(hamming_window)
  LOAD_SYMBOL(hinge_embedding_loss)
  LOAD_SYMBOL(ger)
  LOAD_SYMBOL(ger_out)
  LOAD_SYMBOL(group_norm)
  LOAD_SYMBOL(fft)
  LOAD_SYMBOL(ifft)
  LOAD_SYMBOL(rfft)
  LOAD_SYMBOL(irfft)
  LOAD_SYMBOL(_fft_with_size)
  LOAD_SYMBOL(_cufft_get_plan_cache_size)
  LOAD_SYMBOL(_cufft_get_plan_cache_max_size)
  LOAD_SYMBOL(_cufft_set_plan_cache_max_size)
  LOAD_SYMBOL(_cufft_clear_plan_cache)
  LOAD_SYMBOL(index)
  LOAD_SYMBOL(index_copy_)
  LOAD_SYMBOL(index_copy)
  LOAD_SYMBOL(index_copy_)
  LOAD_SYMBOL(index_copy)
  LOAD_SYMBOL(index_put_)
  LOAD_SYMBOL(index_put)
  LOAD_SYMBOL(_index_put_impl_)
  LOAD_SYMBOL(instance_norm)
  LOAD_SYMBOL(inverse)
  LOAD_SYMBOL(inverse_out)
  LOAD_SYMBOL(_inverse_helper)
  LOAD_SYMBOL(isclose)
  LOAD_SYMBOL(isnan)
  LOAD_SYMBOL(is_distributed)
  LOAD_SYMBOL(is_floating_point)
  LOAD_SYMBOL(is_complex)
  LOAD_SYMBOL(is_nonzero)
  LOAD_SYMBOL(is_same_size)
  LOAD_SYMBOL(is_signed)
  LOAD_SYMBOL(kl_div)
  LOAD_SYMBOL(kl_div_backward)
  LOAD_SYMBOL(kthvalue)
  LOAD_SYMBOL(kthvalue_out)
  LOAD_SYMBOL(kthvalue)
  LOAD_SYMBOL(kthvalue_out)
  LOAD_SYMBOL(layer_norm)
  LOAD_SYMBOL(native_layer_norm)
  LOAD_SYMBOL(native_layer_norm_backward)
  LOAD_SYMBOL(native_layer_norm_double_backward)
  LOAD_SYMBOL(linear)
  LOAD_SYMBOL(mkldnn_linear)
  LOAD_SYMBOL(fbgemm_linear_int8_weight_fp32_activation)
  LOAD_SYMBOL(fbgemm_linear_int8_weight)
  LOAD_SYMBOL(fbgemm_linear_quantize_weight)
  LOAD_SYMBOL(fbgemm_pack_gemm_matrix_fp16)
  LOAD_SYMBOL(fbgemm_linear_fp16_weight_fp32_activation)
  LOAD_SYMBOL(fbgemm_linear_fp16_weight)
  LOAD_SYMBOL(fbgemm_pack_quantized_matrix)
  LOAD_SYMBOL(fbgemm_pack_quantized_matrix)
  LOAD_SYMBOL(linspace)
  LOAD_SYMBOL(linspace_out)
  LOAD_SYMBOL(log)
  LOAD_SYMBOL(log_)
  LOAD_SYMBOL(log_out)
  LOAD_SYMBOL(log10)
  LOAD_SYMBOL(log10_)
  LOAD_SYMBOL(log10_out)
  LOAD_SYMBOL(log1p)
  LOAD_SYMBOL(log1p_)
  LOAD_SYMBOL(log1p_out)
  LOAD_SYMBOL(log2)
  LOAD_SYMBOL(log2_)
  LOAD_SYMBOL(log2_out)
  LOAD_SYMBOL(logdet)
  LOAD_SYMBOL(logspace)
  LOAD_SYMBOL(logspace_out)
  LOAD_SYMBOL(log_softmax)
  LOAD_SYMBOL(log_softmax)
  LOAD_SYMBOL(_log_softmax)
  LOAD_SYMBOL(_log_softmax_backward_data)
  LOAD_SYMBOL(logsumexp)
  LOAD_SYMBOL(logsumexp_out)
  LOAD_SYMBOL(logsumexp)
  LOAD_SYMBOL(logsumexp_out)
  LOAD_SYMBOL(margin_ranking_loss)
  LOAD_SYMBOL(matmul)
  LOAD_SYMBOL(matmul_out)
  LOAD_SYMBOL(matrix_rank)
  LOAD_SYMBOL(matrix_rank)
  LOAD_SYMBOL(matrix_power)
  LOAD_SYMBOL(max)
  LOAD_SYMBOL(max_out)
  LOAD_SYMBOL(max_values)
  LOAD_SYMBOL(max)
  LOAD_SYMBOL(max_out)
  LOAD_SYMBOL(max_values)
  LOAD_SYMBOL(max_pool1d_with_indices)
  LOAD_SYMBOL(max_pool1d)
  LOAD_SYMBOL(max_pool2d)
  LOAD_SYMBOL(mkldnn_max_pool2d)
  LOAD_SYMBOL(quantized_max_pool2d)
  LOAD_SYMBOL(max_pool3d)
  LOAD_SYMBOL(mean)
  LOAD_SYMBOL(mean)
  LOAD_SYMBOL(mean_out)
  LOAD_SYMBOL(mean)
  LOAD_SYMBOL(mean_out)
  LOAD_SYMBOL(median)
  LOAD_SYMBOL(median_out)
  LOAD_SYMBOL(median)
  LOAD_SYMBOL(median_out)
  LOAD_SYMBOL(min)
  LOAD_SYMBOL(min_out)
  LOAD_SYMBOL(min_values)
  LOAD_SYMBOL(min)
  LOAD_SYMBOL(min_out)
  LOAD_SYMBOL(min_values)
  LOAD_SYMBOL(mkldnn_convolution)
  LOAD_SYMBOL(mkldnn_convolution_backward_input)
  LOAD_SYMBOL(mkldnn_convolution_backward_weights)
  LOAD_SYMBOL(mkldnn_convolution_backward)
  LOAD_SYMBOL(miopen_batch_norm)
  LOAD_SYMBOL(miopen_batch_norm_backward)
  LOAD_SYMBOL(miopen_convolution)
  LOAD_SYMBOL(miopen_convolution_backward_input)
  LOAD_SYMBOL(miopen_convolution_backward)
  LOAD_SYMBOL(miopen_convolution_backward_bias)
  LOAD_SYMBOL(miopen_convolution_backward_weight)
  LOAD_SYMBOL(miopen_convolution_transpose)
  LOAD_SYMBOL(miopen_convolution_transpose_backward)
  LOAD_SYMBOL(miopen_convolution_transpose_backward_input)
  LOAD_SYMBOL(miopen_convolution_transpose_backward_weight)
  LOAD_SYMBOL(miopen_depthwise_convolution)
  LOAD_SYMBOL(miopen_depthwise_convolution_backward_input)
  LOAD_SYMBOL(miopen_depthwise_convolution_backward)
  LOAD_SYMBOL(miopen_depthwise_convolution_backward_weight)
  LOAD_SYMBOL(miopen_rnn)
  LOAD_SYMBOL(miopen_rnn_backward)
  LOAD_SYMBOL(mm)
  LOAD_SYMBOL(mm_out)
  LOAD_SYMBOL(_sparse_mm)
  LOAD_SYMBOL(mode)
  LOAD_SYMBOL(mode_out)
  LOAD_SYMBOL(mode)
  LOAD_SYMBOL(mode_out)
  LOAD_SYMBOL(mul)
  LOAD_SYMBOL(mul_)
  LOAD_SYMBOL(mul_out)
  LOAD_SYMBOL(mul)
  LOAD_SYMBOL(mul_)
  LOAD_SYMBOL(mv)
  LOAD_SYMBOL(mv_out)
  LOAD_SYMBOL(mvlgamma)
  LOAD_SYMBOL(mvlgamma_)
  LOAD_SYMBOL(narrow_copy)
  LOAD_SYMBOL(narrow)
  LOAD_SYMBOL(native_batch_norm)
  LOAD_SYMBOL(batch_norm_stats)
  LOAD_SYMBOL(batch_norm_elemt)
  LOAD_SYMBOL(batch_norm_gather_stats)
  LOAD_SYMBOL(batch_norm_gather_stats_with_counts)
  LOAD_SYMBOL(native_batch_norm_backward)
  LOAD_SYMBOL(batch_norm_backward_reduce)
  LOAD_SYMBOL(batch_norm_backward_elemt)
  LOAD_SYMBOL(batch_norm_update_stats)
  LOAD_SYMBOL(_nnpack_available)
  LOAD_SYMBOL(_nnpack_spatial_convolution)
  LOAD_SYMBOL(_nnpack_spatial_convolution_backward)
  LOAD_SYMBOL(_nnpack_spatial_convolution_backward_input)
  LOAD_SYMBOL(_nnpack_spatial_convolution_backward_weight)
  LOAD_SYMBOL(ones)
  LOAD_SYMBOL(ones)
  LOAD_SYMBOL(ones_out)
  LOAD_SYMBOL(ones_like)
  LOAD_SYMBOL(ones_like)
  LOAD_SYMBOL(pairwise_distance)
  LOAD_SYMBOL(cdist)
  LOAD_SYMBOL(_cdist_backward)
  LOAD_SYMBOL(pdist)
  LOAD_SYMBOL(_pdist_forward)
  LOAD_SYMBOL(_pdist_backward)
  LOAD_SYMBOL(cosine_similarity)
  LOAD_SYMBOL(permute)
  LOAD_SYMBOL(numpy_T)
  LOAD_SYMBOL(pixel_shuffle)
  LOAD_SYMBOL(is_pinned)
  LOAD_SYMBOL(pin_memory)
  LOAD_SYMBOL(pinverse)
  LOAD_SYMBOL(poisson_nll_loss)
  LOAD_SYMBOL(scalar_tensor)
  LOAD_SYMBOL(rand)
  LOAD_SYMBOL(rand)
  LOAD_SYMBOL(rand)
  LOAD_SYMBOL(rand)
  LOAD_SYMBOL(rand_out)
  LOAD_SYMBOL(rand_out)
  LOAD_SYMBOL(rand_like)
  LOAD_SYMBOL(rand_like)
  LOAD_SYMBOL(randint)
  LOAD_SYMBOL(randint)
  LOAD_SYMBOL(randint)
  LOAD_SYMBOL(randint)
  LOAD_SYMBOL(randint_out)
  LOAD_SYMBOL(randint_out)
  LOAD_SYMBOL(randint_out)
  LOAD_SYMBOL(randint_out)
  LOAD_SYMBOL(randint_like)
  LOAD_SYMBOL(randint_like)
  LOAD_SYMBOL(randint_like)
  LOAD_SYMBOL(randint_like)
  LOAD_SYMBOL(randn)
  LOAD_SYMBOL(randn)
  LOAD_SYMBOL(randn)
  LOAD_SYMBOL(randn)
  LOAD_SYMBOL(randn_out)
  LOAD_SYMBOL(randn_out)
  LOAD_SYMBOL(randn_like)
  LOAD_SYMBOL(randn_like)
  LOAD_SYMBOL(randperm)
  LOAD_SYMBOL(randperm)
  LOAD_SYMBOL(randperm_out)
  LOAD_SYMBOL(randperm_out)
  LOAD_SYMBOL(range)
  LOAD_SYMBOL(range)
  LOAD_SYMBOL(range_out)
  LOAD_SYMBOL(reciprocal)
  LOAD_SYMBOL(reciprocal_)
  LOAD_SYMBOL(reciprocal_out)
  LOAD_SYMBOL(neg)
  LOAD_SYMBOL(neg_)
  LOAD_SYMBOL(neg_out)
  LOAD_SYMBOL(repeat)
  LOAD_SYMBOL(repeat_interleave)
  LOAD_SYMBOL(repeat_interleave)
  LOAD_SYMBOL(repeat_interleave)
  LOAD_SYMBOL(reshape)
  LOAD_SYMBOL(_mkldnn_reshape)
  LOAD_SYMBOL(reshape_as)
  LOAD_SYMBOL(round)
  LOAD_SYMBOL(round_)
  LOAD_SYMBOL(round_out)
  LOAD_SYMBOL(rrelu)
  LOAD_SYMBOL(rrelu_)
  LOAD_SYMBOL(relu)
  LOAD_SYMBOL(relu_)
  LOAD_SYMBOL(prelu)
  LOAD_SYMBOL(prelu_backward)
  LOAD_SYMBOL(gelu)
  LOAD_SYMBOL(gelu_backward)
  LOAD_SYMBOL(hardshrink)
  LOAD_SYMBOL(hardshrink_backward)
  LOAD_SYMBOL(rsqrt)
  LOAD_SYMBOL(rsqrt_)
  LOAD_SYMBOL(rsqrt_out)
  LOAD_SYMBOL(select)
  LOAD_SYMBOL(select)
  LOAD_SYMBOL(selu)
  LOAD_SYMBOL(selu_)
  LOAD_SYMBOL(celu)
  LOAD_SYMBOL(celu_)
  LOAD_SYMBOL(sigmoid)
  LOAD_SYMBOL(sigmoid_)
  LOAD_SYMBOL(sigmoid_out)
  LOAD_SYMBOL(sin)
  LOAD_SYMBOL(sin_)
  LOAD_SYMBOL(sin_out)
  LOAD_SYMBOL(sinh)
  LOAD_SYMBOL(sinh_)
  LOAD_SYMBOL(sinh_out)
  LOAD_SYMBOL(detach)
  LOAD_SYMBOL(detach_)
  LOAD_SYMBOL(size)
  LOAD_SYMBOL(size)
  LOAD_SYMBOL(slice)
  LOAD_SYMBOL(slogdet)
  LOAD_SYMBOL(smm)
  LOAD_SYMBOL(softmax)
  LOAD_SYMBOL(softmax)
  LOAD_SYMBOL(_softmax)
  LOAD_SYMBOL(_softmax_backward_data)
  LOAD_SYMBOL(split)
  LOAD_SYMBOL(split_with_sizes)
  LOAD_SYMBOL(squeeze)
  LOAD_SYMBOL(squeeze)
  LOAD_SYMBOL(squeeze)
  LOAD_SYMBOL(squeeze_)
  LOAD_SYMBOL(squeeze_)
  LOAD_SYMBOL(squeeze_)
  LOAD_SYMBOL(sspaddmm)
  LOAD_SYMBOL(sspaddmm_out)
  LOAD_SYMBOL(stack)
  LOAD_SYMBOL(stack_out)
  LOAD_SYMBOL(stft)
  LOAD_SYMBOL(stride)
  LOAD_SYMBOL(stride)
  LOAD_SYMBOL(sum)
  LOAD_SYMBOL(sum)
  LOAD_SYMBOL(sum)
  LOAD_SYMBOL(sum_out)
  LOAD_SYMBOL(sum_out)
  LOAD_SYMBOL(sum_to_size)
  LOAD_SYMBOL(sqrt)
  LOAD_SYMBOL(sqrt_)
  LOAD_SYMBOL(sqrt_out)
  LOAD_SYMBOL(std)
  LOAD_SYMBOL(std)
  LOAD_SYMBOL(std_mean)
  LOAD_SYMBOL(std_mean)
  LOAD_SYMBOL(std_mean)
  LOAD_SYMBOL(std_out)
  LOAD_SYMBOL(std)
  LOAD_SYMBOL(std_out)
  LOAD_SYMBOL(prod)
  LOAD_SYMBOL(prod)
  LOAD_SYMBOL(prod_out)
  LOAD_SYMBOL(prod)
  LOAD_SYMBOL(prod_out)
  LOAD_SYMBOL(t)
  LOAD_SYMBOL(t_)
  LOAD_SYMBOL(tan)
  LOAD_SYMBOL(tan_)
  LOAD_SYMBOL(tan_out)
  LOAD_SYMBOL(tanh)
  LOAD_SYMBOL(tanh_)
  LOAD_SYMBOL(tanh_out)
  LOAD_SYMBOL(tensordot)
  LOAD_SYMBOL(threshold)
  LOAD_SYMBOL(threshold_)
  LOAD_SYMBOL(threshold_out)
  LOAD_SYMBOL(threshold_backward)
  LOAD_SYMBOL(transpose)
  LOAD_SYMBOL(transpose)
  LOAD_SYMBOL(_mkldnn_transpose)
  LOAD_SYMBOL(transpose_)
  LOAD_SYMBOL(_mkldnn_transpose_)
  LOAD_SYMBOL(one_hot)
  LOAD_SYMBOL(flip)
  LOAD_SYMBOL(roll)
  LOAD_SYMBOL(rot90)
  LOAD_SYMBOL(trapz)
  LOAD_SYMBOL(trapz)
  LOAD_SYMBOL(_trilinear)
  LOAD_SYMBOL(triplet_margin_loss)
  LOAD_SYMBOL(trunc)
  LOAD_SYMBOL(trunc_)
  LOAD_SYMBOL(trunc_out)
  LOAD_SYMBOL(type_as)
  LOAD_SYMBOL(_has_compatible_shallow_copy_type)
  LOAD_SYMBOL(_unique)
  LOAD_SYMBOL(unique_dim)
  LOAD_SYMBOL(unique_consecutive)
  LOAD_SYMBOL(unique_dim_consecutive)
  LOAD_SYMBOL(_unique2)
  LOAD_SYMBOL(_unsafe_view)
  LOAD_SYMBOL(unsqueeze)
  LOAD_SYMBOL(unsqueeze_)
  LOAD_SYMBOL(var)
  LOAD_SYMBOL(var)
  LOAD_SYMBOL(var_out)
  LOAD_SYMBOL(var)
  LOAD_SYMBOL(var_out)
  LOAD_SYMBOL(var_mean)
  LOAD_SYMBOL(var_mean)
  LOAD_SYMBOL(var_mean)
  LOAD_SYMBOL(view_as)
  LOAD_SYMBOL(where)
  LOAD_SYMBOL(where)
  LOAD_SYMBOL(_s_where)
  LOAD_SYMBOL(norm_except_dim)
  LOAD_SYMBOL(_weight_norm)
  LOAD_SYMBOL(_weight_norm_cuda_interface)
  LOAD_SYMBOL(_weight_norm_cuda_interface_backward)
  LOAD_SYMBOL(_weight_norm_differentiable_backward)
  LOAD_SYMBOL(zeros)
  LOAD_SYMBOL(zeros)
  LOAD_SYMBOL(zeros_out)
  LOAD_SYMBOL(zeros_like)
  LOAD_SYMBOL(zeros_like)
  LOAD_SYMBOL(_standard_gamma_grad)
  LOAD_SYMBOL(_standard_gamma)
  LOAD_SYMBOL(_dirichlet_grad)
  LOAD_SYMBOL(_sample_dirichlet)
  LOAD_SYMBOL(poisson)
  LOAD_SYMBOL(native_norm)
  LOAD_SYMBOL(_sparse_sum)
  LOAD_SYMBOL(_sparse_sum)
  LOAD_SYMBOL(_sparse_sum)
  LOAD_SYMBOL(_sparse_sum)
  LOAD_SYMBOL(_sparse_sum_backward)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm_out)
  LOAD_SYMBOL(norm_out)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm)
  LOAD_SYMBOL(norm_out)
  LOAD_SYMBOL(norm_out)
  LOAD_SYMBOL(frobenius_norm)
  LOAD_SYMBOL(frobenius_norm)
  LOAD_SYMBOL(frobenius_norm_out)
  LOAD_SYMBOL(nuclear_norm)
  LOAD_SYMBOL(nuclear_norm_out)
  LOAD_SYMBOL(nuclear_norm)
  LOAD_SYMBOL(nuclear_norm_out)
  LOAD_SYMBOL(clone)
  LOAD_SYMBOL(resize_as_)
  LOAD_SYMBOL(pow_out)
  LOAD_SYMBOL(pow)
  LOAD_SYMBOL(zero_)
  LOAD_SYMBOL(sub_out)
  LOAD_SYMBOL(sub)
  LOAD_SYMBOL(sub_)
  LOAD_SYMBOL(sub)
  LOAD_SYMBOL(sub_)
  LOAD_SYMBOL(rsub)
  LOAD_SYMBOL(rsub)
  LOAD_SYMBOL(_sparse_addmm)
  LOAD_SYMBOL(addmm_out)
  LOAD_SYMBOL(addmm)
  LOAD_SYMBOL(addmm_)
  LOAD_SYMBOL(sparse_coo_tensor)
  LOAD_SYMBOL(sparse_coo_tensor)
  LOAD_SYMBOL(sparse_coo_tensor)
  LOAD_SYMBOL(_sparse_coo_tensor_unsafe)
  LOAD_SYMBOL(_sparse_coo_tensor_with_dims)
  LOAD_SYMBOL(_sparse_coo_tensor_with_dims_and_tensors)
  LOAD_SYMBOL(sparse_resize_)
  LOAD_SYMBOL(sparse_resize_and_clear_)
  LOAD_SYMBOL(sparse_mask)
  LOAD_SYMBOL(to_dense)
  LOAD_SYMBOL(to_dense_backward)
  LOAD_SYMBOL(sparse_dim)
  LOAD_SYMBOL(_dimI)
  LOAD_SYMBOL(dense_dim)
  LOAD_SYMBOL(_dimV)
  LOAD_SYMBOL(_nnz)
  LOAD_SYMBOL(coalesce)
  LOAD_SYMBOL(is_coalesced)
  LOAD_SYMBOL(_indices)
  LOAD_SYMBOL(_values)
  LOAD_SYMBOL(_coalesced_)
  LOAD_SYMBOL(indices)
  LOAD_SYMBOL(values)
  LOAD_SYMBOL(hspmm_out)
  LOAD_SYMBOL(hspmm)
  LOAD_SYMBOL(copy_sparse_to_sparse_)
  LOAD_SYMBOL(numel)
  LOAD_SYMBOL(unbind)
  LOAD_SYMBOL(unbind)
  LOAD_SYMBOL(to_sparse)
  LOAD_SYMBOL(to_sparse)
  LOAD_SYMBOL(to_mkldnn)
  LOAD_SYMBOL(mkldnn_reorder_conv2d_weight)
  LOAD_SYMBOL(to_mkldnn_backward)
  LOAD_SYMBOL(quantize_per_tensor)
  LOAD_SYMBOL(quantize_per_channel)
  LOAD_SYMBOL(dequantize)
  LOAD_SYMBOL(q_scale)
  LOAD_SYMBOL(q_zero_point)
  LOAD_SYMBOL(q_per_channel_scales)
  LOAD_SYMBOL(q_per_channel_zero_points)
  LOAD_SYMBOL(q_per_channel_axis)
  LOAD_SYMBOL(int_repr)
  LOAD_SYMBOL(_make_per_tensor_quantized_tensor)
  LOAD_SYMBOL(_make_per_channel_quantized_tensor)
  LOAD_SYMBOL(qscheme)
  LOAD_SYMBOL(fake_quantize_per_tensor_affine)
  LOAD_SYMBOL(fake_quantize_per_tensor_affine_backward)
  LOAD_SYMBOL(fake_quantize_per_channel_affine)
  LOAD_SYMBOL(fake_quantize_per_channel_affine_backward)
  LOAD_SYMBOL(to)
  LOAD_SYMBOL(to)
  LOAD_SYMBOL(to)
  LOAD_SYMBOL(to)
  LOAD_SYMBOL(meshgrid)
  LOAD_SYMBOL(cartesian_prod)
  LOAD_SYMBOL(combinations)
  LOAD_SYMBOL(item)
  LOAD_SYMBOL(result_type)
  LOAD_SYMBOL(result_type)
  LOAD_SYMBOL(result_type)
  LOAD_SYMBOL(result_type)
  LOAD_SYMBOL(can_cast)
  LOAD_SYMBOL(promote_types)
  LOAD_SYMBOL(_local_scalar_dense)
  LOAD_SYMBOL(_thnn_fused_lstm_cell)
  LOAD_SYMBOL(_thnn_fused_lstm_cell_backward)
  LOAD_SYMBOL(_thnn_differentiable_lstm_cell_backward)
  LOAD_SYMBOL(_thnn_fused_gru_cell)
  LOAD_SYMBOL(_thnn_fused_gru_cell_backward)
  LOAD_SYMBOL(_thnn_differentiable_gru_cell_backward)
  LOAD_SYMBOL(lstm)
  LOAD_SYMBOL(lstm)
  LOAD_SYMBOL(gru)
  LOAD_SYMBOL(gru)
  LOAD_SYMBOL(rnn_tanh)
  LOAD_SYMBOL(rnn_tanh)
  LOAD_SYMBOL(rnn_relu)
  LOAD_SYMBOL(rnn_relu)
  LOAD_SYMBOL(lstm_cell)
  LOAD_SYMBOL(gru_cell)
  LOAD_SYMBOL(rnn_tanh_cell)
  LOAD_SYMBOL(rnn_relu_cell)
  LOAD_SYMBOL(quantized_lstm)
  LOAD_SYMBOL(quantized_gru)
  LOAD_SYMBOL(quantized_gru)
  LOAD_SYMBOL(quantized_lstm_cell)
  LOAD_SYMBOL(quantized_gru_cell)
  LOAD_SYMBOL(quantized_rnn_relu_cell)
  LOAD_SYMBOL(quantized_rnn_tanh_cell)
  LOAD_SYMBOL(_pack_padded_sequence)
  LOAD_SYMBOL(_pack_padded_sequence_backward)
  LOAD_SYMBOL(_pad_packed_sequence)
  LOAD_SYMBOL(set_)
  LOAD_SYMBOL(set_)
  LOAD_SYMBOL(set_)
  LOAD_SYMBOL(set_)
  LOAD_SYMBOL(set_quantizer_)
  LOAD_SYMBOL(is_set_to)
  LOAD_SYMBOL(masked_fill_)
  LOAD_SYMBOL(masked_fill)
  LOAD_SYMBOL(masked_fill_)
  LOAD_SYMBOL(masked_fill)
  LOAD_SYMBOL(masked_scatter_)
  LOAD_SYMBOL(masked_scatter)
  LOAD_SYMBOL(view)
  LOAD_SYMBOL(put_)
  LOAD_SYMBOL(index_add_)
  LOAD_SYMBOL(index_add)
  LOAD_SYMBOL(index_add)
  LOAD_SYMBOL(index_fill_)
  LOAD_SYMBOL(index_fill)
  LOAD_SYMBOL(index_fill_)
  LOAD_SYMBOL(index_fill)
  LOAD_SYMBOL(index_fill_)
  LOAD_SYMBOL(index_fill_)
  LOAD_SYMBOL(index_fill)
  LOAD_SYMBOL(index_fill)
  LOAD_SYMBOL(scatter_)
  LOAD_SYMBOL(scatter)
  LOAD_SYMBOL(scatter_)
  LOAD_SYMBOL(scatter)
  LOAD_SYMBOL(scatter)
  LOAD_SYMBOL(scatter)
  LOAD_SYMBOL(scatter_add_)
  LOAD_SYMBOL(scatter_add)
  LOAD_SYMBOL(scatter_add)
  LOAD_SYMBOL(lt_)
  LOAD_SYMBOL(lt_)
  LOAD_SYMBOL(gt_)
  LOAD_SYMBOL(gt_)
  LOAD_SYMBOL(le_)
  LOAD_SYMBOL(le_)
  LOAD_SYMBOL(ge_)
  LOAD_SYMBOL(ge_)
  LOAD_SYMBOL(eq_)
  LOAD_SYMBOL(eq_)
  LOAD_SYMBOL(ne_)
  LOAD_SYMBOL(ne_)
  LOAD_SYMBOL(__and__)
  LOAD_SYMBOL(__and__)
  LOAD_SYMBOL(__iand__)
  LOAD_SYMBOL(__iand__)
  LOAD_SYMBOL(__or__)
  LOAD_SYMBOL(__or__)
  LOAD_SYMBOL(__ior__)
  LOAD_SYMBOL(__ior__)
  LOAD_SYMBOL(__xor__)
  LOAD_SYMBOL(__xor__)
  LOAD_SYMBOL(__ixor__)
  LOAD_SYMBOL(__ixor__)
  LOAD_SYMBOL(__lshift__)
  LOAD_SYMBOL(__lshift__)
  LOAD_SYMBOL(__ilshift__)
  LOAD_SYMBOL(__ilshift__)
  LOAD_SYMBOL(__rshift__)
  LOAD_SYMBOL(__rshift__)
  LOAD_SYMBOL(__irshift__)
  LOAD_SYMBOL(__irshift__)
  LOAD_SYMBOL(lgamma_)
  LOAD_SYMBOL(atan2_)
  LOAD_SYMBOL(tril_)
  LOAD_SYMBOL(triu_)
  LOAD_SYMBOL(digamma_)
  LOAD_SYMBOL(polygamma_)
  LOAD_SYMBOL(renorm_)
  LOAD_SYMBOL(pow_)
  LOAD_SYMBOL(pow_)
  LOAD_SYMBOL(lerp_)
  LOAD_SYMBOL(lerp_)
  LOAD_SYMBOL(fmod_)
  LOAD_SYMBOL(fmod_)
  LOAD_SYMBOL(remainder_)
  LOAD_SYMBOL(remainder_)
  LOAD_SYMBOL(addbmm_)
  LOAD_SYMBOL(addbmm_out)
  LOAD_SYMBOL(addbmm)
  LOAD_SYMBOL(addcdiv_)
  LOAD_SYMBOL(random_)
  LOAD_SYMBOL(random_)
  LOAD_SYMBOL(random_)
  LOAD_SYMBOL(uniform_)
  LOAD_SYMBOL(normal_)
  LOAD_SYMBOL(cauchy_)
  LOAD_SYMBOL(log_normal_)
  LOAD_SYMBOL(exponential_)
  LOAD_SYMBOL(geometric_)
  LOAD_SYMBOL(diag_out)
  LOAD_SYMBOL(diag)
  LOAD_SYMBOL(cross_out)
  LOAD_SYMBOL(cross)
  LOAD_SYMBOL(triu_out)
  LOAD_SYMBOL(triu)
  LOAD_SYMBOL(tril_out)
  LOAD_SYMBOL(tril)
  LOAD_SYMBOL(tril_indices)
  LOAD_SYMBOL(triu_indices)
  LOAD_SYMBOL(trace)
  LOAD_SYMBOL(ne_out)
  LOAD_SYMBOL(ne)
  LOAD_SYMBOL(ne_out)
  LOAD_SYMBOL(ne)
  LOAD_SYMBOL(eq_out)
  LOAD_SYMBOL(eq)
  LOAD_SYMBOL(eq_out)
  LOAD_SYMBOL(eq)
  LOAD_SYMBOL(ge_out)
  LOAD_SYMBOL(ge)
  LOAD_SYMBOL(ge_out)
  LOAD_SYMBOL(ge)
  LOAD_SYMBOL(le_out)
  LOAD_SYMBOL(le)
  LOAD_SYMBOL(le_out)
  LOAD_SYMBOL(le)
  LOAD_SYMBOL(gt_out)
  LOAD_SYMBOL(gt)
  LOAD_SYMBOL(gt_out)
  LOAD_SYMBOL(gt)
  LOAD_SYMBOL(lt_out)
  LOAD_SYMBOL(lt)
  LOAD_SYMBOL(lt_out)
  LOAD_SYMBOL(lt)
  LOAD_SYMBOL(take_out)
  LOAD_SYMBOL(take)
  LOAD_SYMBOL(index_select_out)
  LOAD_SYMBOL(index_select)
  LOAD_SYMBOL(index_select_out)
  LOAD_SYMBOL(index_select)
  LOAD_SYMBOL(masked_select_out)
  LOAD_SYMBOL(masked_select)
  LOAD_SYMBOL(nonzero_out)
  LOAD_SYMBOL(nonzero)
  LOAD_SYMBOL(nonzero_numpy)
  LOAD_SYMBOL(gather_out)
  LOAD_SYMBOL(gather)
  LOAD_SYMBOL(gather_out)
  LOAD_SYMBOL(gather)
  LOAD_SYMBOL(_gather_sparse_backward)
  LOAD_SYMBOL(addcmul_out)
  LOAD_SYMBOL(addcmul)
  LOAD_SYMBOL(addcmul_)
  LOAD_SYMBOL(addcdiv_out)
  LOAD_SYMBOL(addcdiv)
  LOAD_SYMBOL(lstsq_out)
  LOAD_SYMBOL(lstsq)
  LOAD_SYMBOL(triangular_solve_out)
  LOAD_SYMBOL(triangular_solve)
  LOAD_SYMBOL(_triangular_solve_helper)
  LOAD_SYMBOL(symeig_out)
  LOAD_SYMBOL(symeig)
  LOAD_SYMBOL(_symeig_helper)
  LOAD_SYMBOL(eig_out)
  LOAD_SYMBOL(eig)
  LOAD_SYMBOL(svd_out)
  LOAD_SYMBOL(svd)
  LOAD_SYMBOL(_svd_helper)
  LOAD_SYMBOL(cholesky_out)
  LOAD_SYMBOL(cholesky)
  LOAD_SYMBOL(_cholesky_helper)
  LOAD_SYMBOL(cholesky_solve_out)
  LOAD_SYMBOL(cholesky_solve)
  LOAD_SYMBOL(_cholesky_solve_helper)
  LOAD_SYMBOL(solve)
  LOAD_SYMBOL(solve_out)
  LOAD_SYMBOL(_solve_helper)
  LOAD_SYMBOL(cholesky_inverse_out)
  LOAD_SYMBOL(cholesky_inverse)
  LOAD_SYMBOL(qr_out)
  LOAD_SYMBOL(qr)
  LOAD_SYMBOL(_qr_helper)
  LOAD_SYMBOL(geqrf_out)
  LOAD_SYMBOL(geqrf)
  LOAD_SYMBOL(orgqr_out)
  LOAD_SYMBOL(orgqr)
  LOAD_SYMBOL(ormqr_out)
  LOAD_SYMBOL(ormqr)
  LOAD_SYMBOL(_lu_with_info)
  LOAD_SYMBOL(lu_solve_out)
  LOAD_SYMBOL(lu_solve)
  LOAD_SYMBOL(_lu_solve_helper)
  LOAD_SYMBOL(multinomial_out)
  LOAD_SYMBOL(multinomial)
  LOAD_SYMBOL(_multinomial_alias_setup)
  LOAD_SYMBOL(_multinomial_alias_draw)
  LOAD_SYMBOL(lgamma_out)
  LOAD_SYMBOL(lgamma)
  LOAD_SYMBOL(digamma_out)
  LOAD_SYMBOL(digamma)
  LOAD_SYMBOL(polygamma_out)
  LOAD_SYMBOL(polygamma)
  LOAD_SYMBOL(erfinv)
  LOAD_SYMBOL(erfinv_)
  LOAD_SYMBOL(erfinv_out)
  LOAD_SYMBOL(sign)
  LOAD_SYMBOL(sign_)
  LOAD_SYMBOL(sign_out)
  LOAD_SYMBOL(dist)
  LOAD_SYMBOL(atan2_out)
  LOAD_SYMBOL(atan2)
  LOAD_SYMBOL(lerp_out)
  LOAD_SYMBOL(lerp_out)
  LOAD_SYMBOL(lerp)
  LOAD_SYMBOL(lerp)
  LOAD_SYMBOL(histc_out)
  LOAD_SYMBOL(histc)
  LOAD_SYMBOL(fmod_out)
  LOAD_SYMBOL(fmod)
  LOAD_SYMBOL(fmod_out)
  LOAD_SYMBOL(fmod)
  LOAD_SYMBOL(remainder_out)
  LOAD_SYMBOL(remainder)
  LOAD_SYMBOL(remainder_out)
  LOAD_SYMBOL(remainder)
  LOAD_SYMBOL(min_out)
  LOAD_SYMBOL(min)
  LOAD_SYMBOL(min)
  LOAD_SYMBOL(max_out)
  LOAD_SYMBOL(max)
  LOAD_SYMBOL(max)
  LOAD_SYMBOL(median)
  LOAD_SYMBOL(sort_out)
  LOAD_SYMBOL(sort)
  LOAD_SYMBOL(sort_out)
  LOAD_SYMBOL(sort)
  LOAD_SYMBOL(argsort)
  LOAD_SYMBOL(argsort)
  LOAD_SYMBOL(topk_out)
  LOAD_SYMBOL(topk)
  LOAD_SYMBOL(all)
  LOAD_SYMBOL(any)
  LOAD_SYMBOL(renorm_out)
  LOAD_SYMBOL(renorm)
  LOAD_SYMBOL(unfold)
  LOAD_SYMBOL(equal)
  LOAD_SYMBOL(pow_out)
  LOAD_SYMBOL(pow)
  LOAD_SYMBOL(pow_out)
  LOAD_SYMBOL(pow)
  LOAD_SYMBOL(normal_out)
  LOAD_SYMBOL(normal)
  LOAD_SYMBOL(normal_out)
  LOAD_SYMBOL(normal)
  LOAD_SYMBOL(normal_out)
  LOAD_SYMBOL(normal)
  LOAD_SYMBOL(normal)
  LOAD_SYMBOL(normal_out)
  LOAD_SYMBOL(alias)
  LOAD_SYMBOL(_addr)
  LOAD_SYMBOL(_addr_)
  LOAD_SYMBOL(_addr_out)
  LOAD_SYMBOL(_index_copy_)
  LOAD_SYMBOL(_cumsum)
  LOAD_SYMBOL(_cumsum_out)
  LOAD_SYMBOL(_cumprod)
  LOAD_SYMBOL(_cumprod_out)
  LOAD_SYMBOL(_var)
  LOAD_SYMBOL(_std)
  LOAD_SYMBOL(_cat)
  LOAD_SYMBOL(_cat_out)
  LOAD_SYMBOL(_mode)
  LOAD_SYMBOL(_mode_out)
  LOAD_SYMBOL(_max)
  LOAD_SYMBOL(_max_out)
  LOAD_SYMBOL(_min)
  LOAD_SYMBOL(_min_out)
  LOAD_SYMBOL(binary_cross_entropy_out)
  LOAD_SYMBOL(binary_cross_entropy)
  LOAD_SYMBOL(binary_cross_entropy_backward_out)
  LOAD_SYMBOL(binary_cross_entropy_backward)
  LOAD_SYMBOL(mse_loss_out)
  LOAD_SYMBOL(mse_loss)
  LOAD_SYMBOL(mse_loss_backward_out)
  LOAD_SYMBOL(mse_loss_backward)
  LOAD_SYMBOL(l1_loss_out)
  LOAD_SYMBOL(l1_loss)
  LOAD_SYMBOL(l1_loss_backward_out)
  LOAD_SYMBOL(l1_loss_backward)
  LOAD_SYMBOL(multi_margin_loss_out)
  LOAD_SYMBOL(multi_margin_loss)
  LOAD_SYMBOL(multi_margin_loss_backward_out)
  LOAD_SYMBOL(multi_margin_loss_backward)
  LOAD_SYMBOL(multilabel_margin_loss_out)
  LOAD_SYMBOL(multilabel_margin_loss)
  LOAD_SYMBOL(multilabel_margin_loss_forward_out)
  LOAD_SYMBOL(multilabel_margin_loss_forward)
  LOAD_SYMBOL(multilabel_margin_loss_backward_out)
  LOAD_SYMBOL(multilabel_margin_loss_backward)
  LOAD_SYMBOL(nll_loss_out)
  LOAD_SYMBOL(nll_loss)
  LOAD_SYMBOL(nll_loss_forward_out)
  LOAD_SYMBOL(nll_loss_forward)
  LOAD_SYMBOL(nll_loss_backward_out)
  LOAD_SYMBOL(nll_loss_backward)
  LOAD_SYMBOL(nll_loss2d_out)
  LOAD_SYMBOL(nll_loss2d)
  LOAD_SYMBOL(nll_loss2d_forward_out)
  LOAD_SYMBOL(nll_loss2d_forward)
  LOAD_SYMBOL(nll_loss2d_backward_out)
  LOAD_SYMBOL(nll_loss2d_backward)
  LOAD_SYMBOL(smooth_l1_loss_out)
  LOAD_SYMBOL(smooth_l1_loss)
  LOAD_SYMBOL(smooth_l1_loss_backward_out)
  LOAD_SYMBOL(smooth_l1_loss_backward)
  LOAD_SYMBOL(soft_margin_loss_out)
  LOAD_SYMBOL(soft_margin_loss)
  LOAD_SYMBOL(soft_margin_loss_backward_out)
  LOAD_SYMBOL(soft_margin_loss_backward)
  LOAD_SYMBOL(elu_out)
  LOAD_SYMBOL(elu)
  LOAD_SYMBOL(elu_backward_out)
  LOAD_SYMBOL(elu_backward)
  LOAD_SYMBOL(elu_)
  LOAD_SYMBOL(glu_out)
  LOAD_SYMBOL(glu)
  LOAD_SYMBOL(glu_backward_out)
  LOAD_SYMBOL(glu_backward)
  LOAD_SYMBOL(hardtanh_out)
  LOAD_SYMBOL(hardtanh)
  LOAD_SYMBOL(hardtanh_backward_out)
  LOAD_SYMBOL(hardtanh_backward)
  LOAD_SYMBOL(hardtanh_)
  LOAD_SYMBOL(leaky_relu_out)
  LOAD_SYMBOL(leaky_relu)
  LOAD_SYMBOL(leaky_relu_backward_out)
  LOAD_SYMBOL(leaky_relu_backward)
  LOAD_SYMBOL(leaky_relu_)
  LOAD_SYMBOL(log_sigmoid_out)
  LOAD_SYMBOL(log_sigmoid)
  LOAD_SYMBOL(log_sigmoid_forward_out)
  LOAD_SYMBOL(log_sigmoid_forward)
  LOAD_SYMBOL(log_sigmoid_backward_out)
  LOAD_SYMBOL(log_sigmoid_backward)
  LOAD_SYMBOL(rrelu_with_noise_out)
  LOAD_SYMBOL(rrelu_with_noise)
  LOAD_SYMBOL(rrelu_with_noise_backward_out)
  LOAD_SYMBOL(rrelu_with_noise_backward)
  LOAD_SYMBOL(rrelu_with_noise_)
  LOAD_SYMBOL(softplus_out)
  LOAD_SYMBOL(softplus)
  LOAD_SYMBOL(softplus_backward_out)
  LOAD_SYMBOL(softplus_backward)
  LOAD_SYMBOL(softshrink_out)
  LOAD_SYMBOL(softshrink)
  LOAD_SYMBOL(softshrink_backward_out)
  LOAD_SYMBOL(softshrink_backward)
  LOAD_SYMBOL(adaptive_avg_pool2d_out)
  LOAD_SYMBOL(adaptive_avg_pool2d)
  LOAD_SYMBOL(mkldnn_adaptive_avg_pool2d)
  LOAD_SYMBOL(_adaptive_avg_pool2d)
  LOAD_SYMBOL(_adaptive_avg_pool2d_backward)
  LOAD_SYMBOL(adaptive_avg_pool3d_out)
  LOAD_SYMBOL(adaptive_avg_pool3d)
  LOAD_SYMBOL(adaptive_avg_pool3d_backward_out)
  LOAD_SYMBOL(adaptive_avg_pool3d_backward)
  LOAD_SYMBOL(adaptive_max_pool2d_out)
  LOAD_SYMBOL(adaptive_max_pool2d)
  LOAD_SYMBOL(adaptive_max_pool2d_backward_out)
  LOAD_SYMBOL(adaptive_max_pool2d_backward)
  LOAD_SYMBOL(adaptive_max_pool3d_out)
  LOAD_SYMBOL(adaptive_max_pool3d)
  LOAD_SYMBOL(adaptive_max_pool3d_backward_out)
  LOAD_SYMBOL(adaptive_max_pool3d_backward)
  LOAD_SYMBOL(avg_pool2d_out)
  LOAD_SYMBOL(avg_pool2d)
  LOAD_SYMBOL(avg_pool2d_backward_out)
  LOAD_SYMBOL(avg_pool2d_backward)
  LOAD_SYMBOL(avg_pool3d_out)
  LOAD_SYMBOL(avg_pool3d)
  LOAD_SYMBOL(avg_pool3d_backward_out)
  LOAD_SYMBOL(avg_pool3d_backward)
  LOAD_SYMBOL(fractional_max_pool2d_out)
  LOAD_SYMBOL(fractional_max_pool2d)
  LOAD_SYMBOL(fractional_max_pool2d_backward_out)
  LOAD_SYMBOL(fractional_max_pool2d_backward)
  LOAD_SYMBOL(fractional_max_pool3d_out)
  LOAD_SYMBOL(fractional_max_pool3d)
  LOAD_SYMBOL(fractional_max_pool3d_backward_out)
  LOAD_SYMBOL(fractional_max_pool3d_backward)
  LOAD_SYMBOL(max_pool2d_with_indices_out)
  LOAD_SYMBOL(max_pool2d_with_indices)
  LOAD_SYMBOL(max_pool2d_with_indices_backward_out)
  LOAD_SYMBOL(max_pool2d_with_indices_backward)
  LOAD_SYMBOL(max_pool3d_with_indices_out)
  LOAD_SYMBOL(max_pool3d_with_indices)
  LOAD_SYMBOL(max_pool3d_with_indices_backward_out)
  LOAD_SYMBOL(max_pool3d_with_indices_backward)
  LOAD_SYMBOL(max_unpool2d_out)
  LOAD_SYMBOL(max_unpool2d)
  LOAD_SYMBOL(max_unpool2d_backward_out)
  LOAD_SYMBOL(max_unpool2d_backward)
  LOAD_SYMBOL(max_unpool3d_out)
  LOAD_SYMBOL(max_unpool3d)
  LOAD_SYMBOL(max_unpool3d_backward_out)
  LOAD_SYMBOL(max_unpool3d_backward)
  LOAD_SYMBOL(reflection_pad1d_out)
  LOAD_SYMBOL(reflection_pad1d)
  LOAD_SYMBOL(reflection_pad1d_backward_out)
  LOAD_SYMBOL(reflection_pad1d_backward)
  LOAD_SYMBOL(reflection_pad2d_out)
  LOAD_SYMBOL(reflection_pad2d)
  LOAD_SYMBOL(reflection_pad2d_backward_out)
  LOAD_SYMBOL(reflection_pad2d_backward)
  LOAD_SYMBOL(replication_pad1d_out)
  LOAD_SYMBOL(replication_pad1d)
  LOAD_SYMBOL(replication_pad1d_backward_out)
  LOAD_SYMBOL(replication_pad1d_backward)
  LOAD_SYMBOL(replication_pad2d_out)
  LOAD_SYMBOL(replication_pad2d)
  LOAD_SYMBOL(replication_pad2d_backward_out)
  LOAD_SYMBOL(replication_pad2d_backward)
  LOAD_SYMBOL(replication_pad3d_out)
  LOAD_SYMBOL(replication_pad3d)
  LOAD_SYMBOL(replication_pad3d_backward_out)
  LOAD_SYMBOL(replication_pad3d_backward)
  LOAD_SYMBOL(upsample_linear1d_out)
  LOAD_SYMBOL(upsample_linear1d)
  LOAD_SYMBOL(upsample_linear1d_backward_out)
  LOAD_SYMBOL(upsample_linear1d_backward)
  LOAD_SYMBOL(upsample_bilinear2d_out)
  LOAD_SYMBOL(upsample_bilinear2d)
  LOAD_SYMBOL(upsample_bilinear2d_backward_out)
  LOAD_SYMBOL(upsample_bilinear2d_backward)
  LOAD_SYMBOL(upsample_bicubic2d_out)
  LOAD_SYMBOL(upsample_bicubic2d)
  LOAD_SYMBOL(upsample_bicubic2d_backward_out)
  LOAD_SYMBOL(upsample_bicubic2d_backward)
  LOAD_SYMBOL(upsample_trilinear3d_out)
  LOAD_SYMBOL(upsample_trilinear3d)
  LOAD_SYMBOL(upsample_trilinear3d_backward_out)
  LOAD_SYMBOL(upsample_trilinear3d_backward)
  LOAD_SYMBOL(upsample_nearest1d_out)
  LOAD_SYMBOL(upsample_nearest1d)
  LOAD_SYMBOL(upsample_nearest1d_backward_out)
  LOAD_SYMBOL(upsample_nearest1d_backward)
  LOAD_SYMBOL(upsample_nearest2d_out)
  LOAD_SYMBOL(upsample_nearest2d)
  LOAD_SYMBOL(upsample_nearest2d_backward_out)
  LOAD_SYMBOL(upsample_nearest2d_backward)
  LOAD_SYMBOL(upsample_nearest3d_out)
  LOAD_SYMBOL(upsample_nearest3d)
  LOAD_SYMBOL(upsample_nearest3d_backward_out)
  LOAD_SYMBOL(upsample_nearest3d_backward)
  LOAD_SYMBOL(sigmoid_backward_out)
  LOAD_SYMBOL(sigmoid_backward)
  LOAD_SYMBOL(tanh_backward_out)
  LOAD_SYMBOL(tanh_backward)
  LOAD_SYMBOL(slow_conv_transpose2d_out)
  LOAD_SYMBOL(slow_conv_transpose2d)
  LOAD_SYMBOL(slow_conv_transpose2d_backward_out)
  LOAD_SYMBOL(slow_conv_transpose2d_backward)
  LOAD_SYMBOL(slow_conv_transpose3d_out)
  LOAD_SYMBOL(slow_conv_transpose3d)
  LOAD_SYMBOL(slow_conv_transpose3d_backward_out)
  LOAD_SYMBOL(slow_conv_transpose3d_backward)
  LOAD_SYMBOL(thnn_conv2d_out)
  LOAD_SYMBOL(thnn_conv2d)
  LOAD_SYMBOL(thnn_conv2d_forward_out)
  LOAD_SYMBOL(thnn_conv2d_forward)
  LOAD_SYMBOL(thnn_conv2d_backward_out)
  LOAD_SYMBOL(thnn_conv2d_backward)
  LOAD_SYMBOL(thnn_conv_depthwise2d_out)
  LOAD_SYMBOL(thnn_conv_depthwise2d)
  LOAD_SYMBOL(thnn_conv_depthwise2d_forward_out)
  LOAD_SYMBOL(thnn_conv_depthwise2d_forward)
  LOAD_SYMBOL(thnn_conv_depthwise2d_backward_out)
  LOAD_SYMBOL(thnn_conv_depthwise2d_backward)
  LOAD_SYMBOL(thnn_conv3d_out)
  LOAD_SYMBOL(thnn_conv3d)
  LOAD_SYMBOL(thnn_conv3d_forward_out)
  LOAD_SYMBOL(thnn_conv3d_forward)
  LOAD_SYMBOL(thnn_conv3d_backward_out)
  LOAD_SYMBOL(thnn_conv3d_backward)
  LOAD_SYMBOL(slow_conv_dilated2d)
  LOAD_SYMBOL(slow_conv_dilated2d_backward)
  LOAD_SYMBOL(slow_conv_dilated3d)
  LOAD_SYMBOL(slow_conv_dilated3d_backward)
  LOAD_SYMBOL(col2im_out)
  LOAD_SYMBOL(col2im)
  LOAD_SYMBOL(col2im_backward_out)
  LOAD_SYMBOL(col2im_backward)
  LOAD_SYMBOL(im2col_out)
  LOAD_SYMBOL(im2col)
  LOAD_SYMBOL(im2col_backward_out)
  LOAD_SYMBOL(im2col_backward)
  */
  /* Autogen Symbols -- End */
  
  return true;
}

#endif
#endif
