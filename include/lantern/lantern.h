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
LANTERN_API void (LANTERN_PTR lantern__cast_byte)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_char)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_double)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_float)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_int)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_long)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_short)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__cast_half)(void* self, const char* selfType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern_backward)(void* self, const char* selfType, void* gradient, const char* gradientType, void* keep_graph, const char* keep_graphType, void* create_graph, const char* create_graphType);
LANTERN_API void (LANTERN_PTR lantern_set_data)(void* self, const char* selfType, void* new_data, const char* new_dataType);
LANTERN_API void (LANTERN_PTR lantern_data)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_leaf)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_output_nr)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__version)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_rename_)(void* self, const char* selfType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern_rename)(void* self, const char* selfType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern_align_to)(void* self, const char* selfType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern_align_as)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_align_tensors)(void* tensors, const char* tensorsType);
LANTERN_API void (LANTERN_PTR lantern_refine_names)(void* self, const char* selfType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern_unflatten)(void* self, const char* selfType, void* dim, const char* dimType, void* sizes, const char* sizesType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern_unflatten)(void* self, const char* selfType, void* dim, const char* dimType, void* sizes, const char* sizesType, void* names, const char* namesType);
LANTERN_API void (LANTERN_PTR lantern__cudnn_ctc_loss)(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* deterministic, const char* deterministicType, void* zero_infinity, const char* zero_infinityType);
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn_flatten_weight)(void* weight_arr, const char* weight_arrType, void* weight_stride0, const char* weight_stride0Type, void* input_size, const char* input_sizeType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn)(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType);
LANTERN_API void (LANTERN_PTR lantern__cudnn_rnn_backward)(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* output, const char* outputType, void* grad_output, const char* grad_outputType, void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType, void* reserve, const char* reserveType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern__cudnn_init_dropout_state)(void* dropout, const char* dropoutType, void* train, const char* trainType, void* dropout_seed, const char* dropout_seedType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__debug_has_internal_overlap)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__fused_dropout)(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern__masked_scale)(void* self, const char* selfType, void* mask, const char* maskType, void* scale, const char* scaleType);
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_draw)(void* quasi, const char* quasiType, void* n, const char* nType, void* sobolstate, const char* sobolstateType, void* dimension, const char* dimensionType, void* num_generated, const char* num_generatedType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_ff_)(void* self, const char* selfType, void* n, const char* nType, void* sobolstate, const char* sobolstateType, void* dimension, const char* dimensionType, void* num_generated, const char* num_generatedType);
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_scramble_)(void* self, const char* selfType, void* ltm, const char* ltmType, void* dimension, const char* dimensionType);
LANTERN_API void (LANTERN_PTR lantern__sobol_engine_initialize_state_)(void* self, const char* selfType, void* dimension, const char* dimensionType);
LANTERN_API void (LANTERN_PTR lantern__reshape_from_tensor)(void* self, const char* selfType, void* shape, const char* shapeType);
LANTERN_API void (LANTERN_PTR lantern__shape_as_tensor)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_dropout)(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_dropout_)(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_feature_dropout)(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_feature_dropout_)(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_alpha_dropout)(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_alpha_dropout_)(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_feature_alpha_dropout)(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_feature_alpha_dropout_)(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType);
LANTERN_API void (LANTERN_PTR lantern_abs)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_abs_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_abs_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_acos)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_acos_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_acos_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool1d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool1d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool1d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_add)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_add_)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_add_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_add)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_add_)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmv)(void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmv_)(void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmv_out)(void* out, const char* outType, void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addr)(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addr_)(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addr_out)(void* out, const char* outType, void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_affine_grid_generator)(void* theta, const char* thetaType, void* size, const char* sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_affine_grid_generator_backward)(void* grad, const char* gradType, void* size, const char* sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_all)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_all_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_all)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_all_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_allclose)(void* self, const char* selfType, void* other, const char* otherType, void* rtol, const char* rtolType, void* atol, const char* atolType, void* equal_nan, const char* equal_nanType);
LANTERN_API void (LANTERN_PTR lantern_any)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_any_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_any)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_any_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_arange)(void* end, const char* endType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_arange)(void* start, const char* startType, void* end, const char* endType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_arange)(void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_arange_out)(void* out, const char* outType, void* end, const char* endType);
LANTERN_API void (LANTERN_PTR lantern_arange_out)(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType);
LANTERN_API void (LANTERN_PTR lantern__dim_arange)(void* like, const char* likeType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_argmax)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_argmin)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_as_strided)(void* self, const char* selfType, void* size, const char* sizeType, void* stride, const char* strideType, void* storage_offset, const char* storage_offsetType);
LANTERN_API void (LANTERN_PTR lantern_as_strided_)(void* self, const char* selfType, void* size, const char* sizeType, void* stride, const char* strideType, void* storage_offset, const char* storage_offsetType);
LANTERN_API void (LANTERN_PTR lantern_asin)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_asin_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_asin_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_atan)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_atan_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_atan_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_baddbmm)(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_baddbmm_)(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern__baddbmm_mkl_)(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_baddbmm_out)(void* out, const char* outType, void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_bartlett_window)(void* window_length, const char* window_lengthType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_bartlett_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType);
LANTERN_API void (LANTERN_PTR lantern__batch_norm_impl_index)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType);
LANTERN_API void (LANTERN_PTR lantern__batch_norm_impl_index_backward)(void* impl_index, const char* impl_indexType, void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var_transform, const char* save_var_transformType, void* train, const char* trainType, void* eps, const char* epsType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_bernoulli)(void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_bernoulli_out)(void* out, const char* outType, void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_bernoulli_)(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_bernoulli_)(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_bernoulli)(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_bilinear)(void* input1, const char* input1Type, void* input2, const char* input2Type, void* weight, const char* weightType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_with_logits)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* pos_weight, const char* pos_weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_with_logits_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* pos_weight, const char* pos_weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_bincount)(void* self, const char* selfType, void* weights, const char* weightsType, void* minlength, const char* minlengthType);
LANTERN_API void (LANTERN_PTR lantern_bitwise_not)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_bitwise_not_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_bitwise_not_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logical_not)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logical_not_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logical_not_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logical_xor)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_logical_xor_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_logical_xor_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_blackman_window)(void* window_length, const char* window_lengthType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_blackman_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_bmm)(void* self, const char* selfType, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_bmm_out)(void* out, const char* outType, void* self, const char* selfType, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_broadcast_tensors)(void* tensors, const char* tensorsType);
LANTERN_API void (LANTERN_PTR lantern_cat)(void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_cat_out)(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_cat)(void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_cat_out)(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_ceil)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_ceil_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_ceil_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_chain_matmul)(void* matrices, const char* matricesType);
LANTERN_API void (LANTERN_PTR lantern_chunk)(void* self, const char* selfType, void* chunks, const char* chunksType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_clamp)(void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_)(void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_out)(void* out, const char* outType, void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_max)(void* self, const char* selfType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_max_)(void* self, const char* selfType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_max_out)(void* out, const char* outType, void* self, const char* selfType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_clamp_min)(void* self, const char* selfType, void* min, const char* minType);
LANTERN_API void (LANTERN_PTR lantern_clamp_min_)(void* self, const char* selfType, void* min, const char* minType);
LANTERN_API void (LANTERN_PTR lantern_clamp_min_out)(void* out, const char* outType, void* self, const char* selfType, void* min, const char* minType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_is_acceptable)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_constant_pad_nd)(void* self, const char* selfType, void* pad, const char* padType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_contiguous)(void* self, const char* selfType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_convolution)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_convolution_overrideable)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_convolution_backward_overrideable)(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* weight, const char* weightType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern__convolution)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* cudnn_enabled, const char* cudnn_enabledType);
LANTERN_API void (LANTERN_PTR lantern__convolution_nogroup)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType);
LANTERN_API void (LANTERN_PTR lantern__convolution_double_backward)(void* ggI, const char* ggIType, void* ggW, const char* ggWType, void* ggb, const char* ggbType, void* gO, const char* gOType, void* weight, const char* weightType, void* self, const char* selfType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* cudnn_enabled, const char* cudnn_enabledType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_conv1d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_conv2d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_conv3d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_conv_tbc)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* pad, const char* padType);
LANTERN_API void (LANTERN_PTR lantern_conv_tbc_backward)(void* self, const char* selfType, void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* pad, const char* padType);
LANTERN_API void (LANTERN_PTR lantern_conv_transpose1d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_conv_transpose2d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_conv_transpose3d)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_copy_)(void* self, const char* selfType, void* src, const char* srcType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern__copy_from)(void* self, const char* selfType, void* dst, const char* dstType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern_cos)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cos_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cos_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cosh)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cosh_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cosh_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_cosine_embedding_loss)(void* input1, const char* input1Type, void* input2, const char* input2Type, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_affine_grid_generator)(void* theta, const char* thetaType, void* N, const char* NType, void* C, const char* CType, void* H, const char* HType, void* W, const char* WType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_affine_grid_generator_backward)(void* grad, const char* gradType, void* N, const char* NType, void* C, const char* CType, void* H, const char* HType, void* W, const char* WType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_batch_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* exponential_average_factor, const char* exponential_average_factorType, void* epsilon, const char* epsilonType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_batch_norm_backward)(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var, const char* save_varType, void* epsilon, const char* epsilonType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_input)(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_bias)(void* grad_output, const char* grad_outputType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_backward_weight)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_bias)(void* grad_output, const char* grad_outputType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_input)(void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_convolution_transpose_backward_weight)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_grid_sampler)(void* self, const char* selfType, void* grid, const char* gridType);
LANTERN_API void (LANTERN_PTR lantern_cudnn_grid_sampler_backward)(void* self, const char* selfType, void* grid, const char* gridType, void* grad_output, const char* grad_outputType);
LANTERN_API void (LANTERN_PTR lantern_cumsum)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumsum_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumsum)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumsum_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumprod)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumprod_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumprod)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_cumprod_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_ctc_loss)(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* reduction, const char* reductionType, void* zero_infinity, const char* zero_infinityType);
LANTERN_API void (LANTERN_PTR lantern_ctc_loss)(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* reduction, const char* reductionType, void* zero_infinity, const char* zero_infinityType);
LANTERN_API void (LANTERN_PTR lantern__ctc_loss)(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* zero_infinity, const char* zero_infinityType);
LANTERN_API void (LANTERN_PTR lantern__ctc_loss_backward)(void* grad, const char* gradType, void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* neg_log_likelihood, const char* neg_log_likelihoodType, void* log_alpha, const char* log_alphaType, void* blank, const char* blankType, void* zero_infinity, const char* zero_infinityType);
LANTERN_API void (LANTERN_PTR lantern_det)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_diag_embed)(void* self, const char* selfType, void* offset, const char* offsetType, void* dim1, const char* dim1Type, void* dim2, const char* dim2Type);
LANTERN_API void (LANTERN_PTR lantern_diagflat)(void* self, const char* selfType, void* offset, const char* offsetType);
LANTERN_API void (LANTERN_PTR lantern_diagonal)(void* self, const char* selfType, void* offset, const char* offsetType, void* dim1, const char* dim1Type, void* dim2, const char* dim2Type);
LANTERN_API void (LANTERN_PTR lantern_fill_diagonal_)(void* self, const char* selfType, void* fill_value, const char* fill_valueType, void* wrap, const char* wrapType);
LANTERN_API void (LANTERN_PTR lantern_div)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_div_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_div_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_div)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_div_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_dot)(void* self, const char* selfType, void* tensor, const char* tensorType);
LANTERN_API void (LANTERN_PTR lantern_dot_out)(void* out, const char* outType, void* self, const char* selfType, void* tensor, const char* tensorType);
LANTERN_API void (LANTERN_PTR lantern_einsum)(void* equation, const char* equationType, void* tensors, const char* tensorsType);
LANTERN_API void (LANTERN_PTR lantern_embedding)(void* weight, const char* weightType, void* indices, const char* indicesType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* sparse, const char* sparseType);
LANTERN_API void (LANTERN_PTR lantern_embedding_backward)(void* grad, const char* gradType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* sparse, const char* sparseType);
LANTERN_API void (LANTERN_PTR lantern_embedding_dense_backward)(void* grad_output, const char* grad_outputType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType);
LANTERN_API void (LANTERN_PTR lantern_embedding_renorm_)(void* self, const char* selfType, void* indices, const char* indicesType, void* max_norm, const char* max_normType, void* norm_type, const char* norm_typeType);
LANTERN_API void (LANTERN_PTR lantern_embedding_sparse_backward)(void* grad, const char* gradType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType);
LANTERN_API void (LANTERN_PTR lantern_embedding_bag)(void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType);
LANTERN_API void (LANTERN_PTR lantern__embedding_bag)(void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType);
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_backward)(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* maximum_indices, const char* maximum_indicesType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType);
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_sparse_backward)(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* per_sample_weights, const char* per_sample_weightsType);
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_dense_backward)(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* maximum_indices, const char* maximum_indicesType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* per_sample_weights, const char* per_sample_weightsType);
LANTERN_API void (LANTERN_PTR lantern__embedding_bag_per_sample_weights_backward)(void* grad, const char* gradType, void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* mode, const char* modeType);
LANTERN_API void (LANTERN_PTR lantern_empty)(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_empty)(void* size, const char* sizeType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_new_empty)(void* self, const char* selfType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_new_full)(void* self, const char* selfType, void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__empty_affine_quantized)(void* size, const char* sizeType, void* options, const char* optionsType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern__empty_per_channel_affine_quantized)(void* size, const char* sizeType, void* scales, const char* scalesType, void* zero_points, const char* zero_pointsType, void* axis, const char* axisType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_resize_)(void* self, const char* selfType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_empty_out)(void* out, const char* outType, void* size, const char* sizeType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_empty_like)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_empty_like)(void* self, const char* selfType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType);
LANTERN_API void (LANTERN_PTR lantern_empty_strided)(void* size, const char* sizeType, void* stride, const char* strideType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_erf)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erf_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erf_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfc)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfc_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfc_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_exp)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_exp_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_exp_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_expm1)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_expm1_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_expm1_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_expand)(void* self, const char* selfType, void* size, const char* sizeType, void* implicit, const char* implicitType);
LANTERN_API void (LANTERN_PTR lantern_expand_as)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eye)(void* n, const char* nType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_eye)(void* n, const char* nType, void* m, const char* mType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_eye_out)(void* out, const char* outType, void* n, const char* nType);
LANTERN_API void (LANTERN_PTR lantern_eye_out)(void* out, const char* outType, void* n, const char* nType, void* m, const char* mType);
LANTERN_API void (LANTERN_PTR lantern_flatten)(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType);
LANTERN_API void (LANTERN_PTR lantern_flatten)(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType, void* out_dim, const char* out_dimType);
LANTERN_API void (LANTERN_PTR lantern_flatten)(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType, void* out_dim, const char* out_dimType);
LANTERN_API void (LANTERN_PTR lantern_flatten)(void* self, const char* selfType, void* dims, const char* dimsType, void* out_dim, const char* out_dimType);
LANTERN_API void (LANTERN_PTR lantern_fill_)(void* self, const char* selfType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_fill_)(void* self, const char* selfType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_floor)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_floor_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_floor_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_frac)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_frac_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_frac_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_full)(void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_full)(void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_full_out)(void* out, const char* outType, void* size, const char* sizeType, void* fill_value, const char* fill_valueType);
LANTERN_API void (LANTERN_PTR lantern_full_like)(void* self, const char* selfType, void* fill_value, const char* fill_valueType);
LANTERN_API void (LANTERN_PTR lantern_full_like)(void* self, const char* selfType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_from_file)(void* filename, const char* filenameType, void* shared, const char* sharedType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_grid_sampler)(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_2d)(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_2d_backward)(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_3d)(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_grid_sampler_3d_backward)(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_hann_window)(void* window_length, const char* window_lengthType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hann_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hamming_window)(void* window_length, const char* window_lengthType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hamming_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hamming_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* alpha, const char* alphaType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hamming_window)(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* alpha, const char* alphaType, void* beta, const char* betaType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_hinge_embedding_loss)(void* self, const char* selfType, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_ger)(void* self, const char* selfType, void* vec2, const char* vec2Type);
LANTERN_API void (LANTERN_PTR lantern_ger_out)(void* out, const char* outType, void* self, const char* selfType, void* vec2, const char* vec2Type);
LANTERN_API void (LANTERN_PTR lantern_group_norm)(void* input, const char* inputType, void* num_groups, const char* num_groupsType, void* weight, const char* weightType, void* bias, const char* biasType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType);
LANTERN_API void (LANTERN_PTR lantern_fft)(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType);
LANTERN_API void (LANTERN_PTR lantern_ifft)(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType);
LANTERN_API void (LANTERN_PTR lantern_rfft)(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType);
LANTERN_API void (LANTERN_PTR lantern_irfft)(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType, void* signal_sizes, const char* signal_sizesType);
LANTERN_API void (LANTERN_PTR lantern__fft_with_size)(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* complex_input, const char* complex_inputType, void* complex_output, const char* complex_outputType, void* inverse, const char* inverseType, void* checked_signal_sizes, const char* checked_signal_sizesType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType, void* output_sizes, const char* output_sizesType);
LANTERN_API void (LANTERN_PTR lantern__cufft_get_plan_cache_size)(void* device_index, const char* device_indexType);
LANTERN_API void (LANTERN_PTR lantern__cufft_get_plan_cache_max_size)(void* device_index, const char* device_indexType);
LANTERN_API void (LANTERN_PTR lantern__cufft_set_plan_cache_max_size)(void* device_index, const char* device_indexType, void* max_size, const char* max_sizeType);
LANTERN_API void (LANTERN_PTR lantern__cufft_clear_plan_cache)(void* device_index, const char* device_indexType);
LANTERN_API void (LANTERN_PTR lantern_index)(void* self, const char* selfType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_index_copy_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_copy)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_copy_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_copy)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_put_)(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType);
LANTERN_API void (LANTERN_PTR lantern_index_put)(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType);
LANTERN_API void (LANTERN_PTR lantern__index_put_impl_)(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType, void* unsafe, const char* unsafeType);
LANTERN_API void (LANTERN_PTR lantern_instance_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* use_input_stats, const char* use_input_statsType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType);
LANTERN_API void (LANTERN_PTR lantern_inverse)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_inverse_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__inverse_helper)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_isclose)(void* self, const char* selfType, void* other, const char* otherType, void* rtol, const char* rtolType, void* atol, const char* atolType, void* equal_nan, const char* equal_nanType);
LANTERN_API void (LANTERN_PTR lantern_isnan)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_distributed)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_floating_point)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_complex)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_nonzero)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_same_size)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_is_signed)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_kl_div)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_kl_div_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_kthvalue)(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_kthvalue_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_kthvalue)(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_kthvalue_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_layer_norm)(void* input, const char* inputType, void* normalized_shape, const char* normalized_shapeType, void* weight, const char* weightType, void* bias, const char* biasType, void* eps, const char* epsType, void* cudnn_enable, const char* cudnn_enableType);
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* M, const char* MType, void* N, const char* NType, void* eps, const char* epsType);
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm_backward)(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* rstd, const char* rstdType, void* weight, const char* weightType, void* M, const char* MType, void* N, const char* NType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_native_layer_norm_double_backward)(void* ggI, const char* ggIType, void* ggW, const char* ggWType, void* ggb, const char* ggbType, void* gO, const char* gOType, void* input, const char* inputType, void* mean, const char* meanType, void* rstd, const char* rstdType, void* weight, const char* weightType, void* M, const char* MType, void* N, const char* NType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_linear)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_linear)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_int8_weight_fp32_activation)(void* input, const char* inputType, void* weight, const char* weightType, void* packed, const char* packedType, void* col_offsets, const char* col_offsetsType, void* weight_scale, const char* weight_scaleType, void* weight_zero_point, const char* weight_zero_pointType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_int8_weight)(void* input, const char* inputType, void* weight, const char* weightType, void* packed, const char* packedType, void* col_offsets, const char* col_offsetsType, void* weight_scale, const char* weight_scaleType, void* weight_zero_point, const char* weight_zero_pointType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_quantize_weight)(void* input, const char* inputType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_gemm_matrix_fp16)(void* input, const char* inputType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_fp16_weight_fp32_activation)(void* input, const char* inputType, void* packed_weight, const char* packed_weightType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_linear_fp16_weight)(void* input, const char* inputType, void* packed_weight, const char* packed_weightType, void* bias, const char* biasType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix)(void* input, const char* inputType);
LANTERN_API void (LANTERN_PTR lantern_fbgemm_pack_quantized_matrix)(void* input, const char* inputType, void* K, const char* KType, void* N, const char* NType);
LANTERN_API void (LANTERN_PTR lantern_linspace)(void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_linspace_out)(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType);
LANTERN_API void (LANTERN_PTR lantern_log)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log10)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log10_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log10_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log1p)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log1p_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log1p_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log2)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log2_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log2_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logdet)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logspace)(void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* base, const char* baseType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_logspace_out)(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* base, const char* baseType);
LANTERN_API void (LANTERN_PTR lantern_log_softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_log_softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern__log_softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* half_to_float, const char* half_to_floatType);
LANTERN_API void (LANTERN_PTR lantern__log_softmax_backward_data)(void* grad_output, const char* grad_outputType, void* output, const char* outputType, void* dim, const char* dimType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_logsumexp)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_logsumexp_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_logsumexp)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_logsumexp_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_margin_ranking_loss)(void* input1, const char* input1Type, void* input2, const char* input2Type, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_matmul)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_matmul_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_matrix_rank)(void* self, const char* selfType, void* tol, const char* tolType, void* symmetric, const char* symmetricType);
LANTERN_API void (LANTERN_PTR lantern_matrix_rank)(void* self, const char* selfType, void* symmetric, const char* symmetricType);
LANTERN_API void (LANTERN_PTR lantern_matrix_power)(void* self, const char* selfType, void* n, const char* nType);
LANTERN_API void (LANTERN_PTR lantern_max)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max_out)(void* max, const char* maxType, void* max_values, const char* max_valuesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max_values)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max_out)(void* max, const char* maxType, void* max_values, const char* max_valuesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max_values)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_max_pool1d_with_indices)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool1d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool2d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_max_pool2d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_quantized_max_pool2d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool3d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_mean)(void* self, const char* selfType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_mean_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_mean_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_median)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_median_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_median)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_median_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min_out)(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min_values)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min_out)(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_min_values)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward_input)(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* bias_defined, const char* bias_definedType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward_weights)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* bias_defined, const char* bias_definedType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_convolution_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_miopen_batch_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* exponential_average_factor, const char* exponential_average_factorType, void* epsilon, const char* epsilonType);
LANTERN_API void (LANTERN_PTR lantern_miopen_batch_norm_backward)(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var, const char* save_varType, void* epsilon, const char* epsilonType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_input)(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_bias)(void* grad_output, const char* grad_outputType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_backward_weight)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward_input)(void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_convolution_transpose_backward_weight)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution)(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_input)(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward)(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_miopen_depthwise_convolution_backward_weight)(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType);
LANTERN_API void (LANTERN_PTR lantern_miopen_rnn)(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* hx, const char* hxType, void* cx, const char* cxType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType);
LANTERN_API void (LANTERN_PTR lantern_miopen_rnn_backward)(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* output, const char* outputType, void* grad_output, const char* grad_outputType, void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType, void* reserve, const char* reserveType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_mm)(void* self, const char* selfType, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_mm_out)(void* out, const char* outType, void* self, const char* selfType, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern__sparse_mm)(void* sparse, const char* sparseType, void* dense, const char* denseType);
LANTERN_API void (LANTERN_PTR lantern_mode)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_mode_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_mode)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_mode_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_mul)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_mul_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_mul_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_mul)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_mul_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_mv)(void* self, const char* selfType, void* vec, const char* vecType);
LANTERN_API void (LANTERN_PTR lantern_mv_out)(void* out, const char* outType, void* self, const char* selfType, void* vec, const char* vecType);
LANTERN_API void (LANTERN_PTR lantern_mvlgamma)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern_mvlgamma_)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern_narrow_copy)(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* length, const char* lengthType);
LANTERN_API void (LANTERN_PTR lantern_narrow)(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* length, const char* lengthType);
LANTERN_API void (LANTERN_PTR lantern_native_batch_norm)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_stats)(void* input, const char* inputType, void* eps, const char* epsType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_elemt)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* eps, const char* epsType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_gather_stats)(void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* count, const char* countType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_gather_stats_with_counts)(void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* counts, const char* countsType);
LANTERN_API void (LANTERN_PTR lantern_native_batch_norm_backward)(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_invstd, const char* save_invstdType, void* train, const char* trainType, void* eps, const char* epsType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_backward_reduce)(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* weight, const char* weightType, void* input_g, const char* input_gType, void* weight_g, const char* weight_gType, void* bias_g, const char* bias_gType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_backward_elemt)(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* weight, const char* weightType, void* mean_dy, const char* mean_dyType, void* mean_dy_xmu, const char* mean_dy_xmuType);
LANTERN_API void (LANTERN_PTR lantern_batch_norm_update_stats)(void* input, const char* inputType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType);
LANTERN_API void (LANTERN_PTR lantern__nnpack_available)();
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution)(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward)(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_input)(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern__nnpack_spatial_convolution_backward_weight)(void* input, const char* inputType, void* weightsize, const char* weightsizeType, void* grad_output, const char* grad_outputType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_ones)(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_ones)(void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_ones_out)(void* out, const char* outType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_ones_like)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_ones_like)(void* self, const char* selfType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_pairwise_distance)(void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType, void* eps, const char* epsType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_cdist)(void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern__cdist_backward)(void* grad, const char* gradType, void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType, void* cdist, const char* cdistType);
LANTERN_API void (LANTERN_PTR lantern_pdist)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern__pdist_forward)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern__pdist_backward)(void* grad, const char* gradType, void* self, const char* selfType, void* p, const char* pType, void* pdist, const char* pdistType);
LANTERN_API void (LANTERN_PTR lantern_cosine_similarity)(void* x1, const char* x1Type, void* x2, const char* x2Type, void* dim, const char* dimType, void* eps, const char* epsType);
LANTERN_API void (LANTERN_PTR lantern_permute)(void* self, const char* selfType, void* dims, const char* dimsType);
LANTERN_API void (LANTERN_PTR lantern_numpy_t)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_pixel_shuffle)(void* self, const char* selfType, void* upscale_factor, const char* upscale_factorType);
LANTERN_API void (LANTERN_PTR lantern_is_pinned)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_pin_memory)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_pinverse)(void* self, const char* selfType, void* rcond, const char* rcondType);
LANTERN_API void (LANTERN_PTR lantern_poisson_nll_loss)(void* input, const char* inputType, void* target, const char* targetType, void* log_input, const char* log_inputType, void* full, const char* fullType, void* eps, const char* epsType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_scalar_tensor)(void* s, const char* sType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_rand)(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_rand)(void* size, const char* sizeType, void* generator, const char* generatorType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_rand)(void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_rand)(void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_rand_out)(void* out, const char* outType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_rand_out)(void* out, const char* outType, void* size, const char* sizeType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_rand_like)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_rand_like)(void* self, const char* selfType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint)(void* high, const char* highType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint)(void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint)(void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint)(void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint_out)(void* out, const char* outType, void* high, const char* highType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_randint_out)(void* out, const char* outType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_randint_out)(void* out, const char* outType, void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_randint_out)(void* out, const char* outType, void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_randint_like)(void* self, const char* selfType, void* high, const char* highType);
LANTERN_API void (LANTERN_PTR lantern_randint_like)(void* self, const char* selfType, void* low, const char* lowType, void* high, const char* highType);
LANTERN_API void (LANTERN_PTR lantern_randint_like)(void* self, const char* selfType, void* high, const char* highType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randint_like)(void* self, const char* selfType, void* low, const char* lowType, void* high, const char* highType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randn)(void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randn)(void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randn)(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randn)(void* size, const char* sizeType, void* generator, const char* generatorType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randn_out)(void* out, const char* outType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_randn_out)(void* out, const char* outType, void* size, const char* sizeType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_randn_like)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_randn_like)(void* self, const char* selfType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randperm)(void* n, const char* nType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randperm)(void* n, const char* nType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_randperm_out)(void* out, const char* outType, void* n, const char* nType);
LANTERN_API void (LANTERN_PTR lantern_randperm_out)(void* out, const char* outType, void* n, const char* nType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_range)(void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_range)(void* start, const char* startType, void* end, const char* endType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_range_out)(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType);
LANTERN_API void (LANTERN_PTR lantern_reciprocal)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_reciprocal_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_reciprocal_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_neg)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_neg_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_neg_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_repeat)(void* self, const char* selfType, void* repeats, const char* repeatsType);
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)(void* repeats, const char* repeatsType);
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)(void* self, const char* selfType, void* repeats, const char* repeatsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_repeat_interleave)(void* self, const char* selfType, void* repeats, const char* repeatsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_reshape)(void* self, const char* selfType, void* shape, const char* shapeType);
LANTERN_API void (LANTERN_PTR lantern__mkldnn_reshape)(void* self, const char* selfType, void* shape, const char* shapeType);
LANTERN_API void (LANTERN_PTR lantern_reshape_as)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_round)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_round_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_round_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_rrelu)(void* self, const char* selfType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_)(void* self, const char* selfType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_relu)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_relu_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_prelu)(void* self, const char* selfType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_prelu_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_gelu)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_gelu_backward)(void* grad, const char* gradType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_hardshrink)(void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_hardshrink_backward)(void* grad_out, const char* grad_outType, void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_rsqrt)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_rsqrt_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_rsqrt_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_select)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_select)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_selu)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_selu_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_celu)(void* self, const char* selfType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_celu_)(void* self, const char* selfType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sigmoid)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sigmoid_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sigmoid_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sin)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sin_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sin_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sinh)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sinh_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sinh_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_detach)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_detach_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_size)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_size)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_slice)(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType);
LANTERN_API void (LANTERN_PTR lantern_slogdet)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_smm)(void* self, const char* selfType, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern__softmax)(void* self, const char* selfType, void* dim, const char* dimType, void* half_to_float, const char* half_to_floatType);
LANTERN_API void (LANTERN_PTR lantern__softmax_backward_data)(void* grad_output, const char* grad_outputType, void* output, const char* outputType, void* dim, const char* dimType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_split)(void* self, const char* selfType, void* split_size, const char* split_sizeType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_split_with_sizes)(void* self, const char* selfType, void* split_sizes, const char* split_sizesType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_squeeze)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_squeeze)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_squeeze)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_squeeze_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_squeeze_)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_squeeze_)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_sspaddmm)(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sspaddmm_out)(void* out, const char* outType, void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_stack)(void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_stack_out)(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_stft)(void* self, const char* selfType, void* n_fft, const char* n_fftType, void* hop_length, const char* hop_lengthType, void* win_length, const char* win_lengthType, void* window, const char* windowType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType);
LANTERN_API void (LANTERN_PTR lantern_stride)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_stride)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_sum)(void* self, const char* selfType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_sum)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_sum)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_sum_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_sum_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_sum_to_size)(void* self, const char* selfType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_sqrt)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sqrt_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sqrt_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_std)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern_std)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_std_mean)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern_std_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_std_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_std_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_std)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_std_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_prod)(void* self, const char* selfType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_prod)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_prod_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_prod)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_prod_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_t)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_t_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tan)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tan_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tan_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tanh)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tanh_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tanh_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_tensordot)(void* self, const char* selfType, void* other, const char* otherType, void* dims_self, const char* dims_selfType, void* dims_other, const char* dims_otherType);
LANTERN_API void (LANTERN_PTR lantern_threshold)(void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_threshold_)(void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_threshold_out)(void* out, const char* outType, void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_threshold_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* threshold, const char* thresholdType);
LANTERN_API void (LANTERN_PTR lantern_transpose)(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type);
LANTERN_API void (LANTERN_PTR lantern_transpose)(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type);
LANTERN_API void (LANTERN_PTR lantern__mkldnn_transpose)(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type);
LANTERN_API void (LANTERN_PTR lantern_transpose_)(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type);
LANTERN_API void (LANTERN_PTR lantern__mkldnn_transpose_)(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type);
LANTERN_API void (LANTERN_PTR lantern_one_hot)(void* self, const char* selfType, void* num_classes, const char* num_classesType);
LANTERN_API void (LANTERN_PTR lantern_flip)(void* self, const char* selfType, void* dims, const char* dimsType);
LANTERN_API void (LANTERN_PTR lantern_roll)(void* self, const char* selfType, void* shifts, const char* shiftsType, void* dims, const char* dimsType);
LANTERN_API void (LANTERN_PTR lantern_rot90)(void* self, const char* selfType, void* k, const char* kType, void* dims, const char* dimsType);
LANTERN_API void (LANTERN_PTR lantern_trapz)(void* y, const char* yType, void* x, const char* xType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_trapz)(void* y, const char* yType, void* dx, const char* dxType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__trilinear)(void* i1, const char* i1Type, void* i2, const char* i2Type, void* i3, const char* i3Type, void* expand1, const char* expand1Type, void* expand2, const char* expand2Type, void* expand3, const char* expand3Type, void* sumdim, const char* sumdimType, void* unroll_dim, const char* unroll_dimType);
LANTERN_API void (LANTERN_PTR lantern_triplet_margin_loss)(void* anchor, const char* anchorType, void* positive, const char* positiveType, void* negative, const char* negativeType, void* margin, const char* marginType, void* p, const char* pType, void* eps, const char* epsType, void* swap, const char* swapType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_trunc)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_trunc_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_trunc_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_type_as)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern__has_compatible_shallow_copy_type)(void* self, const char* selfType, void* from, const char* fromType);
LANTERN_API void (LANTERN_PTR lantern__unique)(void* self, const char* selfType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType);
LANTERN_API void (LANTERN_PTR lantern_unique_dim)(void* self, const char* selfType, void* dim, const char* dimType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType);
LANTERN_API void (LANTERN_PTR lantern_unique_consecutive)(void* self, const char* selfType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_unique_dim_consecutive)(void* self, const char* selfType, void* dim, const char* dimType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType);
LANTERN_API void (LANTERN_PTR lantern__unique2)(void* self, const char* selfType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType);
LANTERN_API void (LANTERN_PTR lantern__unsafe_view)(void* self, const char* selfType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_unsqueeze)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_unsqueeze_)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_var)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern_var)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_var_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_var)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_var_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_var_mean)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern_var_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_var_mean)(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_view_as)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_where)(void* condition, const char* conditionType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_where)(void* condition, const char* conditionType);
LANTERN_API void (LANTERN_PTR lantern__s_where)(void* condition, const char* conditionType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_norm_except_dim)(void* v, const char* vType, void* pow, const char* powType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__weight_norm)(void* v, const char* vType, void* g, const char* gType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__weight_norm_cuda_interface)(void* v, const char* vType, void* g, const char* gType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__weight_norm_cuda_interface_backward)(void* grad_w, const char* grad_wType, void* saved_v, const char* saved_vType, void* saved_g, const char* saved_gType, void* saved_norms, const char* saved_normsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__weight_norm_differentiable_backward)(void* grad_w, const char* grad_wType, void* saved_v, const char* saved_vType, void* saved_g, const char* saved_gType, void* saved_norms, const char* saved_normsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_zeros)(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_zeros)(void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_zeros_out)(void* out, const char* outType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_zeros_like)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_zeros_like)(void* self, const char* selfType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__standard_gamma_grad)(void* self, const char* selfType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern__standard_gamma)(void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern__dirichlet_grad)(void* x, const char* xType, void* alpha, const char* alphaType, void* total, const char* totalType);
LANTERN_API void (LANTERN_PTR lantern__sample_dirichlet)(void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_poisson)(void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_native_norm)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)(void* self, const char* selfType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__sparse_sum)(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern__sparse_sum_backward)(void* grad, const char* gradType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_norm)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_frobenius_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm)(void* self, const char* selfType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_nuclear_norm_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_clone)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_resize_as_)(void* self, const char* selfType, void* the_template, const char* the_templateType);
LANTERN_API void (LANTERN_PTR lantern_pow_out)(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_pow)(void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_zero_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sub_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sub)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sub_)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sub)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sub_)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_rsub)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_rsub)(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern__sparse_addmm)(void* self, const char* selfType, void* sparse, const char* sparseType, void* dense, const char* denseType, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmm_out)(void* out, const char* outType, void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmm)(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addmm_)(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)(void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)(void* indices, const char* indicesType, void* values, const char* valuesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_sparse_coo_tensor)(void* indices, const char* indicesType, void* values, const char* valuesType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_unsafe)(void* indices, const char* indicesType, void* values, const char* valuesType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_with_dims)(void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType, void* size, const char* sizeType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern__sparse_coo_tensor_with_dims_and_tensors)(void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType, void* size, const char* sizeType, void* indices, const char* indicesType, void* values, const char* valuesType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_sparse_resize_)(void* self, const char* selfType, void* size, const char* sizeType, void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType);
LANTERN_API void (LANTERN_PTR lantern_sparse_resize_and_clear_)(void* self, const char* selfType, void* size, const char* sizeType, void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType);
LANTERN_API void (LANTERN_PTR lantern_sparse_mask)(void* self, const char* selfType, void* mask, const char* maskType);
LANTERN_API void (LANTERN_PTR lantern_to_dense)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_to_dense_backward)(void* grad, const char* gradType, void* input, const char* inputType);
LANTERN_API void (LANTERN_PTR lantern_sparse_dim)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__dimi)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_dense_dim)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__dimv)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__nnz)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_coalesce)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_is_coalesced)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__indices)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__values)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__coalesced_)(void* self, const char* selfType, void* coalesced, const char* coalescedType);
LANTERN_API void (LANTERN_PTR lantern_indices)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_values)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_hspmm_out)(void* out, const char* outType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_hspmm)(void* mat1, const char* mat1Type, void* mat2, const char* mat2Type);
LANTERN_API void (LANTERN_PTR lantern_copy_sparse_to_sparse_)(void* self, const char* selfType, void* src, const char* srcType, void* non_blocking, const char* non_blockingType);
LANTERN_API void (LANTERN_PTR lantern_numel)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_unbind)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_unbind)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_to_sparse)(void* self, const char* selfType, void* sparse_dim, const char* sparse_dimType);
LANTERN_API void (LANTERN_PTR lantern_to_sparse)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_to_mkldnn)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_reorder_conv2d_weight)(void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType);
LANTERN_API void (LANTERN_PTR lantern_to_mkldnn_backward)(void* grad, const char* gradType, void* input, const char* inputType);
LANTERN_API void (LANTERN_PTR lantern_quantize_per_tensor)(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_quantize_per_channel)(void* self, const char* selfType, void* scales, const char* scalesType, void* zero_points, const char* zero_pointsType, void* axis, const char* axisType, void* dtype, const char* dtypeType);
LANTERN_API void (LANTERN_PTR lantern_dequantize)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_q_scale)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_q_zero_point)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_scales)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_zero_points)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_q_per_channel_axis)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_int_repr)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__make_per_tensor_quantized_tensor)(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType);
LANTERN_API void (LANTERN_PTR lantern__make_per_channel_quantized_tensor)(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType);
LANTERN_API void (LANTERN_PTR lantern_qscheme)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_tensor_affine)(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType);
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_tensor_affine_backward)(void* grad, const char* gradType, void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType);
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_channel_affine)(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType);
LANTERN_API void (LANTERN_PTR lantern_fake_quantize_per_channel_affine_backward)(void* grad, const char* gradType, void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType);
LANTERN_API void (LANTERN_PTR lantern_to)(void* self, const char* selfType, void* options, const char* optionsType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType);
LANTERN_API void (LANTERN_PTR lantern_to)(void* self, const char* selfType, void* device, const char* deviceType, void* dtype, const char* dtypeType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType);
LANTERN_API void (LANTERN_PTR lantern_to)(void* self, const char* selfType, void* dtype, const char* dtypeType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType);
LANTERN_API void (LANTERN_PTR lantern_to)(void* self, const char* selfType, void* other, const char* otherType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType);
LANTERN_API void (LANTERN_PTR lantern_meshgrid)(void* tensors, const char* tensorsType);
LANTERN_API void (LANTERN_PTR lantern_cartesian_prod)(void* tensors, const char* tensorsType);
LANTERN_API void (LANTERN_PTR lantern_combinations)(void* self, const char* selfType, void* r, const char* rType, void* with_replacement, const char* with_replacementType);
LANTERN_API void (LANTERN_PTR lantern_item)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_result_type)(void* tensor, const char* tensorType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_result_type)(void* tensor, const char* tensorType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_result_type)(void* scalar, const char* scalarType, void* tensor, const char* tensorType);
LANTERN_API void (LANTERN_PTR lantern_result_type)(void* scalar1, const char* scalar1Type, void* scalar2, const char* scalar2Type);
LANTERN_API void (LANTERN_PTR lantern_can_cast)(void* from, const char* fromType, void* to, const char* toType);
LANTERN_API void (LANTERN_PTR lantern_promote_types)(void* type1, const char* type1Type, void* type2, const char* type2Type);
LANTERN_API void (LANTERN_PTR lantern__local_scalar_dense)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_lstm_cell)(void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* cx, const char* cxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType);
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_lstm_cell_backward)(void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* cx, const char* cxType, void* cy, const char* cyType, void* workspace, const char* workspaceType, void* has_bias, const char* has_biasType);
LANTERN_API void (LANTERN_PTR lantern__thnn_differentiable_lstm_cell_backward)(void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType, void* cx, const char* cxType, void* cy, const char* cyType);
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_gru_cell)(void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* hx, const char* hxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType);
LANTERN_API void (LANTERN_PTR lantern__thnn_fused_gru_cell_backward)(void* grad_hy, const char* grad_hyType, void* workspace, const char* workspaceType, void* has_bias, const char* has_biasType);
LANTERN_API void (LANTERN_PTR lantern__thnn_differentiable_gru_cell_backward)(void* grad_hy, const char* grad_hyType, void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* hx, const char* hxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType);
LANTERN_API void (LANTERN_PTR lantern_lstm)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern_lstm)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern_gru)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern_gru)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern_rnn_relu)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern_rnn_relu)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern_lstm_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType);
LANTERN_API void (LANTERN_PTR lantern_gru_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType);
LANTERN_API void (LANTERN_PTR lantern_rnn_tanh_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType);
LANTERN_API void (LANTERN_PTR lantern_rnn_relu_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType);
LANTERN_API void (LANTERN_PTR lantern_quantized_lstm)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType, void* dtype, const char* dtypeType, void* use_dynamic, const char* use_dynamicType);
LANTERN_API void (LANTERN_PTR lantern_quantized_gru)(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern_quantized_gru)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType);
LANTERN_API void (LANTERN_PTR lantern_quantized_lstm_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType);
LANTERN_API void (LANTERN_PTR lantern_quantized_gru_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType);
LANTERN_API void (LANTERN_PTR lantern_quantized_rnn_relu_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType);
LANTERN_API void (LANTERN_PTR lantern_quantized_rnn_tanh_cell)(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType);
LANTERN_API void (LANTERN_PTR lantern__pack_padded_sequence)(void* input, const char* inputType, void* lengths, const char* lengthsType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern__pack_padded_sequence_backward)(void* grad, const char* gradType, void* input_size, const char* input_sizeType, void* batch_sizes, const char* batch_sizesType, void* batch_first, const char* batch_firstType);
LANTERN_API void (LANTERN_PTR lantern__pad_packed_sequence)(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* batch_first, const char* batch_firstType, void* padding_value, const char* padding_valueType, void* total_length, const char* total_lengthType);
LANTERN_API void (LANTERN_PTR lantern_set_)(void* self, const char* selfType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_set_)(void* self, const char* selfType, void* source, const char* sourceType, void* storage_offset, const char* storage_offsetType, void* size, const char* sizeType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_set_)(void* self, const char* selfType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_set_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_set_quantizer_)(void* self, const char* selfType, void* quantizer, const char* quantizerType);
LANTERN_API void (LANTERN_PTR lantern_is_set_to)(void* self, const char* selfType, void* tensor, const char* tensorType);
LANTERN_API void (LANTERN_PTR lantern_masked_fill_)(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_masked_fill)(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_masked_fill_)(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_masked_fill)(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_masked_scatter_)(void* self, const char* selfType, void* mask, const char* maskType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_masked_scatter)(void* self, const char* selfType, void* mask, const char* maskType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_view)(void* self, const char* selfType, void* size, const char* sizeType);
LANTERN_API void (LANTERN_PTR lantern_put_)(void* self, const char* selfType, void* index, const char* indexType, void* source, const char* sourceType, void* accumulate, const char* accumulateType);
LANTERN_API void (LANTERN_PTR lantern_index_add_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_add)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_add)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern_index_fill_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_index_fill)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_scatter_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_scatter)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_scatter_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_scatter)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_scatter)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_scatter)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_scatter_add_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_scatter_add)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_scatter_add)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType);
LANTERN_API void (LANTERN_PTR lantern_lt_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lt_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ne_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ne_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___and__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___and__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___iand__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___iand__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___or__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___or__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ior__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ior__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___xor__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___xor__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ixor__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ixor__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___lshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___lshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ilshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___ilshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___rshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___rshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___irshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern___irshift__)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lgamma_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_atan2_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_tril_)(void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_triu_)(void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_digamma_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_polygamma_)(void* self, const char* selfType, void* n, const char* nType);
LANTERN_API void (LANTERN_PTR lantern_renorm_)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType);
LANTERN_API void (LANTERN_PTR lantern_pow_)(void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_pow_)(void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_lerp_)(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_lerp_)(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_fmod_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_fmod_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder_)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_addbmm_)(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addbmm_out)(void* out, const char* outType, void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addbmm)(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern_addcdiv_)(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_random_)(void* self, const char* selfType, void* from, const char* fromType, void* to, const char* toType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_random_)(void* self, const char* selfType, void* to, const char* toType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_random_)(void* self, const char* selfType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_uniform_)(void* self, const char* selfType, void* from, const char* fromType, void* to, const char* toType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal_)(void* self, const char* selfType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_cauchy_)(void* self, const char* selfType, void* median, const char* medianType, void* sigma, const char* sigmaType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_log_normal_)(void* self, const char* selfType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_exponential_)(void* self, const char* selfType, void* lambd, const char* lambdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_geometric_)(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_diag_out)(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_diag)(void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_cross_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_cross)(void* self, const char* selfType, void* other, const char* otherType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_triu_out)(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_triu)(void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_tril_out)(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_tril)(void* self, const char* selfType, void* diagonal, const char* diagonalType);
LANTERN_API void (LANTERN_PTR lantern_tril_indices)(void* row, const char* rowType, void* col, const char* colType, void* offset, const char* offsetType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_triu_indices)(void* row, const char* rowType, void* col, const char* colType, void* offset, const char* offsetType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_trace)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_ne_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ne)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ne_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ne)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_eq)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_ge)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_le)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_gt)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lt_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lt)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lt_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lt)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_take_out)(void* out, const char* outType, void* self, const char* selfType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_take)(void* self, const char* selfType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_index_select_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_index_select)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_index_select_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_index_select)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType);
LANTERN_API void (LANTERN_PTR lantern_masked_select_out)(void* out, const char* outType, void* self, const char* selfType, void* mask, const char* maskType);
LANTERN_API void (LANTERN_PTR lantern_masked_select)(void* self, const char* selfType, void* mask, const char* maskType);
LANTERN_API void (LANTERN_PTR lantern_nonzero_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_nonzero)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_nonzero_numpy)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_gather_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType);
LANTERN_API void (LANTERN_PTR lantern_gather)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType);
LANTERN_API void (LANTERN_PTR lantern_gather_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType);
LANTERN_API void (LANTERN_PTR lantern_gather)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType);
LANTERN_API void (LANTERN_PTR lantern__gather_sparse_backward)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* grad, const char* gradType);
LANTERN_API void (LANTERN_PTR lantern_addcmul_out)(void* out, const char* outType, void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_addcmul)(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_addcmul_)(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_addcdiv_out)(void* out, const char* outType, void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_addcdiv)(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType);
LANTERN_API void (LANTERN_PTR lantern_lstsq_out)(void* X, const char* XType, void* qr, const char* qrType, void* self, const char* selfType, void* A, const char* AType);
LANTERN_API void (LANTERN_PTR lantern_lstsq)(void* self, const char* selfType, void* A, const char* AType);
LANTERN_API void (LANTERN_PTR lantern_triangular_solve_out)(void* X, const char* XType, void* M, const char* MType, void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType);
LANTERN_API void (LANTERN_PTR lantern_triangular_solve)(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType);
LANTERN_API void (LANTERN_PTR lantern__triangular_solve_helper)(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType);
LANTERN_API void (LANTERN_PTR lantern_symeig_out)(void* e, const char* eType, void* V, const char* VType, void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_symeig)(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern__symeig_helper)(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_eig_out)(void* e, const char* eType, void* v, const char* vType, void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType);
LANTERN_API void (LANTERN_PTR lantern_eig)(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType);
LANTERN_API void (LANTERN_PTR lantern_svd_out)(void* U, const char* UType, void* S, const char* SType, void* V, const char* VType, void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType);
LANTERN_API void (LANTERN_PTR lantern_svd)(void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType);
LANTERN_API void (LANTERN_PTR lantern__svd_helper)(void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType);
LANTERN_API void (LANTERN_PTR lantern_cholesky_out)(void* out, const char* outType, void* self, const char* selfType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_cholesky)(void* self, const char* selfType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern__cholesky_helper)(void* self, const char* selfType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_cholesky_solve_out)(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_cholesky_solve)(void* self, const char* selfType, void* input2, const char* input2Type, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern__cholesky_solve_helper)(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_solve)(void* self, const char* selfType, void* A, const char* AType);
LANTERN_API void (LANTERN_PTR lantern_solve_out)(void* solution, const char* solutionType, void* lu, const char* luType, void* self, const char* selfType, void* A, const char* AType);
LANTERN_API void (LANTERN_PTR lantern__solve_helper)(void* self, const char* selfType, void* A, const char* AType);
LANTERN_API void (LANTERN_PTR lantern_cholesky_inverse_out)(void* out, const char* outType, void* self, const char* selfType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_cholesky_inverse)(void* self, const char* selfType, void* upper, const char* upperType);
LANTERN_API void (LANTERN_PTR lantern_qr_out)(void* Q, const char* QType, void* R, const char* RType, void* self, const char* selfType, void* some, const char* someType);
LANTERN_API void (LANTERN_PTR lantern_qr)(void* self, const char* selfType, void* some, const char* someType);
LANTERN_API void (LANTERN_PTR lantern__qr_helper)(void* self, const char* selfType, void* some, const char* someType);
LANTERN_API void (LANTERN_PTR lantern_geqrf_out)(void* a, const char* aType, void* tau, const char* tauType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_geqrf)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_orgqr_out)(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type);
LANTERN_API void (LANTERN_PTR lantern_orgqr)(void* self, const char* selfType, void* input2, const char* input2Type);
LANTERN_API void (LANTERN_PTR lantern_ormqr_out)(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type, void* input3, const char* input3Type, void* left, const char* leftType, void* transpose, const char* transposeType);
LANTERN_API void (LANTERN_PTR lantern_ormqr)(void* self, const char* selfType, void* input2, const char* input2Type, void* input3, const char* input3Type, void* left, const char* leftType, void* transpose, const char* transposeType);
LANTERN_API void (LANTERN_PTR lantern__lu_with_info)(void* self, const char* selfType, void* pivot, const char* pivotType, void* check_errors, const char* check_errorsType);
LANTERN_API void (LANTERN_PTR lantern_lu_solve_out)(void* out, const char* outType, void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType);
LANTERN_API void (LANTERN_PTR lantern_lu_solve)(void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType);
LANTERN_API void (LANTERN_PTR lantern__lu_solve_helper)(void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType);
LANTERN_API void (LANTERN_PTR lantern_multinomial_out)(void* out, const char* outType, void* self, const char* selfType, void* num_samples, const char* num_samplesType, void* replacement, const char* replacementType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_multinomial)(void* self, const char* selfType, void* num_samples, const char* num_samplesType, void* replacement, const char* replacementType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern__multinomial_alias_setup)(void* probs, const char* probsType);
LANTERN_API void (LANTERN_PTR lantern__multinomial_alias_draw)(void* J, const char* JType, void* q, const char* qType, void* num_samples, const char* num_samplesType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_lgamma_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_lgamma)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_digamma_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_digamma)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_polygamma_out)(void* out, const char* outType, void* n, const char* nType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_polygamma)(void* n, const char* nType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfinv)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfinv_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_erfinv_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sign)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sign_)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sign_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_dist)(void* self, const char* selfType, void* other, const char* otherType, void* p, const char* pType);
LANTERN_API void (LANTERN_PTR lantern_atan2_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_atan2)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_lerp_out)(void* out, const char* outType, void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_lerp_out)(void* out, const char* outType, void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_lerp)(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_lerp)(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType);
LANTERN_API void (LANTERN_PTR lantern_histc_out)(void* out, const char* outType, void* self, const char* selfType, void* bins, const char* binsType, void* min, const char* minType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_histc)(void* self, const char* selfType, void* bins, const char* binsType, void* min, const char* minType, void* max, const char* maxType);
LANTERN_API void (LANTERN_PTR lantern_fmod_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_fmod)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_fmod_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_fmod)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_remainder)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_min_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_min)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_min)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_max_out)(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_max)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_max)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_median)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_sort_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_sort)(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_sort_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_sort)(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_argsort)(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_argsort)(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType);
LANTERN_API void (LANTERN_PTR lantern_topk_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* largest, const char* largestType, void* sorted, const char* sortedType);
LANTERN_API void (LANTERN_PTR lantern_topk)(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* largest, const char* largestType, void* sorted, const char* sortedType);
LANTERN_API void (LANTERN_PTR lantern_all)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_any)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_renorm_out)(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType);
LANTERN_API void (LANTERN_PTR lantern_renorm)(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType);
LANTERN_API void (LANTERN_PTR lantern_unfold)(void* self, const char* selfType, void* dimension, const char* dimensionType, void* size, const char* sizeType, void* step, const char* stepType);
LANTERN_API void (LANTERN_PTR lantern_equal)(void* self, const char* selfType, void* other, const char* otherType);
LANTERN_API void (LANTERN_PTR lantern_pow_out)(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_pow)(void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_pow_out)(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_pow)(void* self, const char* selfType, void* exponent, const char* exponentType);
LANTERN_API void (LANTERN_PTR lantern_normal_out)(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal)(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal_out)(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal)(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal_out)(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal)(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_normal)(void* mean, const char* meanType, void* std, const char* stdType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType);
LANTERN_API void (LANTERN_PTR lantern_normal_out)(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* size, const char* sizeType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_alias)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern__addr)(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern__addr_)(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern__addr_out)(void* out, const char* outType, void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType);
LANTERN_API void (LANTERN_PTR lantern__index_copy_)(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType);
LANTERN_API void (LANTERN_PTR lantern__cumsum)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__cumsum_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__cumprod)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__cumprod_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__var)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern__std)(void* self, const char* selfType, void* unbiased, const char* unbiasedType);
LANTERN_API void (LANTERN_PTR lantern__cat)(void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__cat_out)(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern__mode)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern__mode_out)(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern__max)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern__max_out)(void* max, const char* maxType, void* max_indices, const char* max_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern__min)(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern__min_out)(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_binary_cross_entropy_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_mse_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_mse_loss)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_mse_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_mse_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_l1_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_l1_loss)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_l1_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_l1_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss)(void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multi_margin_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_forward_out)(void* output, const char* outputType, void* is_target, const char* is_targetType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_forward)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType, void* is_target, const char* is_targetType);
LANTERN_API void (LANTERN_PTR lantern_multilabel_margin_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType, void* is_target, const char* is_targetType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss_forward_out)(void* output, const char* outputType, void* total_weight, const char* total_weightType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss_forward)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_forward_out)(void* output, const char* outputType, void* total_weight, const char* total_weightType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_forward)(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType);
LANTERN_API void (LANTERN_PTR lantern_nll_loss2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType);
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_smooth_l1_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_out)(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss)(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_soft_margin_loss_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType);
LANTERN_API void (LANTERN_PTR lantern_elu_out)(void* out, const char* outType, void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType);
LANTERN_API void (LANTERN_PTR lantern_elu)(void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType);
LANTERN_API void (LANTERN_PTR lantern_elu_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_elu_backward)(void* grad_output, const char* grad_outputType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_elu_)(void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType);
LANTERN_API void (LANTERN_PTR lantern_glu_out)(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_glu)(void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_glu_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_glu_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* dim, const char* dimType);
LANTERN_API void (LANTERN_PTR lantern_hardtanh_out)(void* out, const char* outType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType);
LANTERN_API void (LANTERN_PTR lantern_hardtanh)(void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType);
LANTERN_API void (LANTERN_PTR lantern_hardtanh_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType);
LANTERN_API void (LANTERN_PTR lantern_hardtanh_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType);
LANTERN_API void (LANTERN_PTR lantern_hardtanh_)(void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType);
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_out)(void* out, const char* outType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType);
LANTERN_API void (LANTERN_PTR lantern_leaky_relu)(void* self, const char* selfType, void* negative_slope, const char* negative_slopeType);
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType);
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType);
LANTERN_API void (LANTERN_PTR lantern_leaky_relu_)(void* self, const char* selfType, void* negative_slope, const char* negative_slopeType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_out)(void* out, const char* outType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_forward_out)(void* output, const char* outputType, void* buffer, const char* bufferType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_forward)(void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* buffer, const char* bufferType);
LANTERN_API void (LANTERN_PTR lantern_log_sigmoid_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* buffer, const char* bufferType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_out)(void* out, const char* outType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise)(void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType);
LANTERN_API void (LANTERN_PTR lantern_rrelu_with_noise_)(void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType);
LANTERN_API void (LANTERN_PTR lantern_softplus_out)(void* out, const char* outType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType);
LANTERN_API void (LANTERN_PTR lantern_softplus)(void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType);
LANTERN_API void (LANTERN_PTR lantern_softplus_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_softplus_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_softshrink_out)(void* out, const char* outType, void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_softshrink)(void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_softshrink_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_softshrink_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* lambd, const char* lambdType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool2d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_mkldnn_adaptive_avg_pool2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern__adaptive_avg_pool2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern__adaptive_avg_pool2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_avg_pool3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_out)(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_out)(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_adaptive_max_pool3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_out)(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_out)(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_avg_pool3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_out)(void* output, const char* outputType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_out)(void* output, const char* outputType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_fractional_max_pool3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_out)(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_max_pool2d_with_indices_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_out)(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType);
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_max_pool3d_with_indices_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_out)(void* out, const char* outType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d)(void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_out)(void* out, const char* outType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d)(void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_max_unpool3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_out)(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d)(void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad1d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_out)(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d)(void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_reflection_pad2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_out)(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d)(void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad1d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_out)(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d)(void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_out)(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d)(void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_replication_pad3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d)(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_linear1d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bilinear2d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_bicubic2d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d)(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_trilinear3d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest1d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest2d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d)(void* self, const char* selfType, void* output_size, const char* output_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_upsample_nearest3d_backward)(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType);
LANTERN_API void (LANTERN_PTR lantern_sigmoid_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_sigmoid_backward)(void* grad_output, const char* grad_outputType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_tanh_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_tanh_backward)(void* grad_output, const char* grad_outputType, void* output, const char* outputType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* columns, const char* columnsType, void* ones, const char* onesType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* columns, const char* columnsType, void* ones, const char* onesType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_transpose3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_forward_out)(void* output, const char* outputType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_forward)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_forward)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv_depthwise2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_out)(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_forward_out)(void* output, const char* outputType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_forward)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_backward_out)(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType);
LANTERN_API void (LANTERN_PTR lantern_thnn_conv3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated2d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated2d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated3d)(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType);
LANTERN_API void (LANTERN_PTR lantern_slow_conv_dilated3d_backward)(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType);
LANTERN_API void (LANTERN_PTR lantern_col2im_out)(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_col2im)(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_col2im_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_col2im_backward)(void* grad_output, const char* grad_outputType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_im2col_out)(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_im2col)(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_im2col_backward_out)(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* input_size, const char* input_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
LANTERN_API void (LANTERN_PTR lantern_im2col_backward)(void* grad_output, const char* grad_outputType, void* input_size, const char* input_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType);
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
