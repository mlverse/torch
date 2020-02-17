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
/*
void lantern__cast_byte(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_char(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_double(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_float(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_int(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_long(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_short(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern__cast_half(void* self, const char* selfType, void* non_blocking, const char* non_blockingType) {}
void lantern_backward(void* self, const char* selfType, void* gradient, const char* gradientType, void* keep_graph, const char* keep_graphType, void* create_graph, const char* create_graphType) {}
void lantern_set_data(void* self, const char* selfType, void* new_data, const char* new_dataType) {}
void lantern_data(void* self, const char* selfType) {}
void lantern_is_leaf(void* self, const char* selfType) {}
void lantern_output_nr(void* self, const char* selfType) {}
void lantern__version(void* self, const char* selfType) {}
void lantern_rename_(void* self, const char* selfType, void* names, const char* namesType) {}
void lantern_rename(void* self, const char* selfType, void* names, const char* namesType) {}
void lantern_align_to(void* self, const char* selfType, void* names, const char* namesType) {}
void lantern_align_as(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_align_tensors(void* tensors, const char* tensorsType) {}
void lantern_refine_names(void* self, const char* selfType, void* names, const char* namesType) {}
void lantern_unflatten(void* self, const char* selfType, void* dim, const char* dimType, void* sizes, const char* sizesType, void* names, const char* namesType) {}
void lantern_unflatten(void* self, const char* selfType, void* dim, const char* dimType, void* sizes, const char* sizesType, void* names, const char* namesType) {}
void lantern__cudnn_ctc_loss(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* deterministic, const char* deterministicType, void* zero_infinity, const char* zero_infinityType) {}
void lantern__cudnn_rnn_flatten_weight(void* weight_arr, const char* weight_arrType, void* weight_stride0, const char* weight_stride0Type, void* input_size, const char* input_sizeType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* bidirectional, const char* bidirectionalType) {}
void lantern__cudnn_rnn(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType) {}
void lantern__cudnn_rnn_backward(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* output, const char* outputType, void* grad_output, const char* grad_outputType, void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType, void* reserve, const char* reserveType, void* output_mask, const char* output_maskType) {}
void lantern__cudnn_init_dropout_state(void* dropout, const char* dropoutType, void* train, const char* trainType, void* dropout_seed, const char* dropout_seedType, void* options, const char* optionsType) {}
void lantern__debug_has_internal_overlap(void* self, const char* selfType) {}
void lantern__fused_dropout(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType) {}
void lantern__masked_scale(void* self, const char* selfType, void* mask, const char* maskType, void* scale, const char* scaleType) {}
void lantern__sobol_engine_draw(void* quasi, const char* quasiType, void* n, const char* nType, void* sobolstate, const char* sobolstateType, void* dimension, const char* dimensionType, void* num_generated, const char* num_generatedType, void* dtype, const char* dtypeType) {}
void lantern__sobol_engine_ff_(void* self, const char* selfType, void* n, const char* nType, void* sobolstate, const char* sobolstateType, void* dimension, const char* dimensionType, void* num_generated, const char* num_generatedType) {}
void lantern__sobol_engine_scramble_(void* self, const char* selfType, void* ltm, const char* ltmType, void* dimension, const char* dimensionType) {}
void lantern__sobol_engine_initialize_state_(void* self, const char* selfType, void* dimension, const char* dimensionType) {}
void lantern__reshape_from_tensor(void* self, const char* selfType, void* shape, const char* shapeType) {}
void lantern__shape_as_tensor(void* self, const char* selfType) {}
void lantern_dropout(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_dropout_(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_feature_dropout(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_feature_dropout_(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_alpha_dropout(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_alpha_dropout_(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_feature_alpha_dropout(void* input, const char* inputType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_feature_alpha_dropout_(void* self, const char* selfType, void* p, const char* pType, void* train, const char* trainType) {}
void lantern_abs(void* self, const char* selfType) {}
void lantern_abs_(void* self, const char* selfType) {}
void lantern_abs_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_acos(void* self, const char* selfType) {}
void lantern_acos_(void* self, const char* selfType) {}
void lantern_acos_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_avg_pool1d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType) {}
void lantern_adaptive_avg_pool1d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_max_pool1d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_add(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_add_(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_add_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_add(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_add_(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_addmv(void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addmv_(void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addmv_out(void* out, const char* outType, void* self, const char* selfType, void* mat, const char* matType, void* vec, const char* vecType, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addr(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addr_(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addr_out(void* out, const char* outType, void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_affine_grid_generator(void* theta, const char* thetaType, void* size, const char* sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_affine_grid_generator_backward(void* grad, const char* gradType, void* size, const char* sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_all(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_all_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_all(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_all_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_allclose(void* self, const char* selfType, void* other, const char* otherType, void* rtol, const char* rtolType, void* atol, const char* atolType, void* equal_nan, const char* equal_nanType) {}
void lantern_any(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_any_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_any(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_any_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_arange(void* end, const char* endType, void* options, const char* optionsType) {}
void lantern_arange(void* start, const char* startType, void* end, const char* endType, void* options, const char* optionsType) {}
void lantern_arange(void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType, void* options, const char* optionsType) {}
void lantern_arange_out(void* out, const char* outType, void* end, const char* endType) {}
void lantern_arange_out(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType) {}
void lantern__dim_arange(void* like, const char* likeType, void* dim, const char* dimType) {}
void lantern_argmax(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_argmin(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_as_strided(void* self, const char* selfType, void* size, const char* sizeType, void* stride, const char* strideType, void* storage_offset, const char* storage_offsetType) {}
void lantern_as_strided_(void* self, const char* selfType, void* size, const char* sizeType, void* stride, const char* strideType, void* storage_offset, const char* storage_offsetType) {}
void lantern_asin(void* self, const char* selfType) {}
void lantern_asin_(void* self, const char* selfType) {}
void lantern_asin_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_atan(void* self, const char* selfType) {}
void lantern_atan_(void* self, const char* selfType) {}
void lantern_atan_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_baddbmm(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_baddbmm_(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern__baddbmm_mkl_(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_baddbmm_out(void* out, const char* outType, void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_bartlett_window(void* window_length, const char* window_lengthType, void* options, const char* optionsType) {}
void lantern_bartlett_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType) {}
void lantern_batch_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType) {}
void lantern__batch_norm_impl_index(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType) {}
void lantern__batch_norm_impl_index_backward(void* impl_index, const char* impl_indexType, void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var_transform, const char* save_var_transformType, void* train, const char* trainType, void* eps, const char* epsType, void* output_mask, const char* output_maskType) {}
void lantern_bernoulli(void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern_bernoulli_out(void* out, const char* outType, void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern_bernoulli_(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType) {}
void lantern_bernoulli_(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType) {}
void lantern_bernoulli(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType) {}
void lantern_bilinear(void* input1, const char* input1Type, void* input2, const char* input2Type, void* weight, const char* weightType, void* bias, const char* biasType) {}
void lantern_binary_cross_entropy_with_logits(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* pos_weight, const char* pos_weightType, void* reduction, const char* reductionType) {}
void lantern_binary_cross_entropy_with_logits_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* pos_weight, const char* pos_weightType, void* reduction, const char* reductionType) {}
void lantern_bincount(void* self, const char* selfType, void* weights, const char* weightsType, void* minlength, const char* minlengthType) {}
void lantern_bitwise_not(void* self, const char* selfType) {}
void lantern_bitwise_not_(void* self, const char* selfType) {}
void lantern_bitwise_not_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_logical_not(void* self, const char* selfType) {}
void lantern_logical_not_(void* self, const char* selfType) {}
void lantern_logical_not_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_logical_xor(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_logical_xor_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_logical_xor_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_blackman_window(void* window_length, const char* window_lengthType, void* options, const char* optionsType) {}
void lantern_blackman_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType) {}
void lantern_bmm(void* self, const char* selfType, void* mat2, const char* mat2Type) {}
void lantern_bmm_out(void* out, const char* outType, void* self, const char* selfType, void* mat2, const char* mat2Type) {}
void lantern_broadcast_tensors(void* tensors, const char* tensorsType) {}
void lantern_cat(void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_cat_out(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_cat(void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_cat_out(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_ceil(void* self, const char* selfType) {}
void lantern_ceil_(void* self, const char* selfType) {}
void lantern_ceil_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_chain_matmul(void* matrices, const char* matricesType) {}
void lantern_chunk(void* self, const char* selfType, void* chunks, const char* chunksType, void* dim, const char* dimType) {}
void lantern_clamp(void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType) {}
void lantern_clamp_(void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType) {}
void lantern_clamp_out(void* out, const char* outType, void* self, const char* selfType, void* min, const char* minType, void* max, const char* maxType) {}
void lantern_clamp_max(void* self, const char* selfType, void* max, const char* maxType) {}
void lantern_clamp_max_(void* self, const char* selfType, void* max, const char* maxType) {}
void lantern_clamp_max_out(void* out, const char* outType, void* self, const char* selfType, void* max, const char* maxType) {}
void lantern_clamp_min(void* self, const char* selfType, void* min, const char* minType) {}
void lantern_clamp_min_(void* self, const char* selfType, void* min, const char* minType) {}
void lantern_clamp_min_out(void* out, const char* outType, void* self, const char* selfType, void* min, const char* minType) {}
void lantern_cudnn_is_acceptable(void* self, const char* selfType) {}
void lantern_constant_pad_nd(void* self, const char* selfType, void* pad, const char* padType, void* value, const char* valueType) {}
void lantern_contiguous(void* self, const char* selfType, void* memory_format, const char* memory_formatType) {}
void lantern_convolution(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType) {}
void lantern_convolution_overrideable(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType) {}
void lantern_convolution_backward_overrideable(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* weight, const char* weightType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* output_mask, const char* output_maskType) {}
void lantern__convolution(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* cudnn_enabled, const char* cudnn_enabledType) {}
void lantern__convolution_nogroup(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType) {}
void lantern__convolution_double_backward(void* ggI, const char* ggIType, void* ggW, const char* ggWType, void* ggb, const char* ggbType, void* gO, const char* gOType, void* weight, const char* weightType, void* self, const char* selfType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* transposed, const char* transposedType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* cudnn_enabled, const char* cudnn_enabledType, void* output_mask, const char* output_maskType) {}
void lantern_conv1d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType) {}
void lantern_conv2d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType) {}
void lantern_conv3d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* groups, const char* groupsType) {}
void lantern_conv_tbc(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* pad, const char* padType) {}
void lantern_conv_tbc_backward(void* self, const char* selfType, void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* pad, const char* padType) {}
void lantern_conv_transpose1d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType) {}
void lantern_conv_transpose2d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType) {}
void lantern_conv_transpose3d(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* groups, const char* groupsType, void* dilation, const char* dilationType) {}
void lantern_copy_(void* self, const char* selfType, void* src, const char* srcType, void* non_blocking, const char* non_blockingType) {}
void lantern__copy_from(void* self, const char* selfType, void* dst, const char* dstType, void* non_blocking, const char* non_blockingType) {}
void lantern_cos(void* self, const char* selfType) {}
void lantern_cos_(void* self, const char* selfType) {}
void lantern_cos_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_cosh(void* self, const char* selfType) {}
void lantern_cosh_(void* self, const char* selfType) {}
void lantern_cosh_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_cosine_embedding_loss(void* input1, const char* input1Type, void* input2, const char* input2Type, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType) {}
void lantern_cudnn_affine_grid_generator(void* theta, const char* thetaType, void* N, const char* NType, void* C, const char* CType, void* H, const char* HType, void* W, const char* WType) {}
void lantern_cudnn_affine_grid_generator_backward(void* grad, const char* gradType, void* N, const char* NType, void* C, const char* CType, void* H, const char* HType, void* W, const char* WType) {}
void lantern_cudnn_batch_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* exponential_average_factor, const char* exponential_average_factorType, void* epsilon, const char* epsilonType) {}
void lantern_cudnn_batch_norm_backward(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var, const char* save_varType, void* epsilon, const char* epsilonType) {}
void lantern_cudnn_convolution(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_convolution_backward_input(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_convolution_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType) {}
void lantern_cudnn_convolution_backward_bias(void* grad_output, const char* grad_outputType) {}
void lantern_cudnn_convolution_backward_weight(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_convolution_transpose(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_convolution_transpose_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType) {}
void lantern_cudnn_convolution_transpose_backward_bias(void* grad_output, const char* grad_outputType) {}
void lantern_cudnn_convolution_transpose_backward_input(void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_convolution_transpose_backward_weight(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_cudnn_grid_sampler(void* self, const char* selfType, void* grid, const char* gridType) {}
void lantern_cudnn_grid_sampler_backward(void* self, const char* selfType, void* grid, const char* gridType, void* grad_output, const char* grad_outputType) {}
void lantern_cumsum(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumsum_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumsum(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumsum_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumprod(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumprod_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumprod(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_cumprod_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_ctc_loss(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* reduction, const char* reductionType, void* zero_infinity, const char* zero_infinityType) {}
void lantern_ctc_loss(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* reduction, const char* reductionType, void* zero_infinity, const char* zero_infinityType) {}
void lantern__ctc_loss(void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* blank, const char* blankType, void* zero_infinity, const char* zero_infinityType) {}
void lantern__ctc_loss_backward(void* grad, const char* gradType, void* log_probs, const char* log_probsType, void* targets, const char* targetsType, void* input_lengths, const char* input_lengthsType, void* target_lengths, const char* target_lengthsType, void* neg_log_likelihood, const char* neg_log_likelihoodType, void* log_alpha, const char* log_alphaType, void* blank, const char* blankType, void* zero_infinity, const char* zero_infinityType) {}
void lantern_det(void* self, const char* selfType) {}
void lantern_diag_embed(void* self, const char* selfType, void* offset, const char* offsetType, void* dim1, const char* dim1Type, void* dim2, const char* dim2Type) {}
void lantern_diagflat(void* self, const char* selfType, void* offset, const char* offsetType) {}
void lantern_diagonal(void* self, const char* selfType, void* offset, const char* offsetType, void* dim1, const char* dim1Type, void* dim2, const char* dim2Type) {}
void lantern_fill_diagonal_(void* self, const char* selfType, void* fill_value, const char* fill_valueType, void* wrap, const char* wrapType) {}
void lantern_div(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_div_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_div_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_div(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_div_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_dot(void* self, const char* selfType, void* tensor, const char* tensorType) {}
void lantern_dot_out(void* out, const char* outType, void* self, const char* selfType, void* tensor, const char* tensorType) {}
void lantern_einsum(void* equation, const char* equationType, void* tensors, const char* tensorsType) {}
void lantern_embedding(void* weight, const char* weightType, void* indices, const char* indicesType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* sparse, const char* sparseType) {}
void lantern_embedding_backward(void* grad, const char* gradType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* sparse, const char* sparseType) {}
void lantern_embedding_dense_backward(void* grad_output, const char* grad_outputType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType) {}
void lantern_embedding_renorm_(void* self, const char* selfType, void* indices, const char* indicesType, void* max_norm, const char* max_normType, void* norm_type, const char* norm_typeType) {}
void lantern_embedding_sparse_backward(void* grad, const char* gradType, void* indices, const char* indicesType, void* num_weights, const char* num_weightsType, void* padding_idx, const char* padding_idxType, void* scale_grad_by_freq, const char* scale_grad_by_freqType) {}
void lantern_embedding_bag(void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType) {}
void lantern__embedding_bag(void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType) {}
void lantern__embedding_bag_backward(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* maximum_indices, const char* maximum_indicesType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* sparse, const char* sparseType, void* per_sample_weights, const char* per_sample_weightsType) {}
void lantern__embedding_bag_sparse_backward(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* per_sample_weights, const char* per_sample_weightsType) {}
void lantern__embedding_bag_dense_backward(void* grad, const char* gradType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* bag_size, const char* bag_sizeType, void* maximum_indices, const char* maximum_indicesType, void* num_weights, const char* num_weightsType, void* scale_grad_by_freq, const char* scale_grad_by_freqType, void* mode, const char* modeType, void* per_sample_weights, const char* per_sample_weightsType) {}
void lantern__embedding_bag_per_sample_weights_backward(void* grad, const char* gradType, void* weight, const char* weightType, void* indices, const char* indicesType, void* offsets, const char* offsetsType, void* offset2bag, const char* offset2bagType, void* mode, const char* modeType) {}
void lantern_empty(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType) {}
void lantern_empty(void* size, const char* sizeType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType) {}
void lantern_new_empty(void* self, const char* selfType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_new_full(void* self, const char* selfType, void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType) {}
void lantern__empty_affine_quantized(void* size, const char* sizeType, void* options, const char* optionsType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* memory_format, const char* memory_formatType) {}
void lantern__empty_per_channel_affine_quantized(void* size, const char* sizeType, void* scales, const char* scalesType, void* zero_points, const char* zero_pointsType, void* axis, const char* axisType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType) {}
void lantern_resize_(void* self, const char* selfType, void* size, const char* sizeType) {}
void lantern_empty_out(void* out, const char* outType, void* size, const char* sizeType, void* memory_format, const char* memory_formatType) {}
void lantern_empty_like(void* self, const char* selfType) {}
void lantern_empty_like(void* self, const char* selfType, void* options, const char* optionsType, void* memory_format, const char* memory_formatType) {}
void lantern_empty_strided(void* size, const char* sizeType, void* stride, const char* strideType, void* options, const char* optionsType) {}
void lantern_erf(void* self, const char* selfType) {}
void lantern_erf_(void* self, const char* selfType) {}
void lantern_erf_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_erfc(void* self, const char* selfType) {}
void lantern_erfc_(void* self, const char* selfType) {}
void lantern_erfc_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_exp(void* self, const char* selfType) {}
void lantern_exp_(void* self, const char* selfType) {}
void lantern_exp_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_expm1(void* self, const char* selfType) {}
void lantern_expm1_(void* self, const char* selfType) {}
void lantern_expm1_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_expand(void* self, const char* selfType, void* size, const char* sizeType, void* implicit, const char* implicitType) {}
void lantern_expand_as(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eye(void* n, const char* nType, void* options, const char* optionsType) {}
void lantern_eye(void* n, const char* nType, void* m, const char* mType, void* options, const char* optionsType) {}
void lantern_eye_out(void* out, const char* outType, void* n, const char* nType) {}
void lantern_eye_out(void* out, const char* outType, void* n, const char* nType, void* m, const char* mType) {}
void lantern_flatten(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType) {}
void lantern_flatten(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType, void* out_dim, const char* out_dimType) {}
void lantern_flatten(void* self, const char* selfType, void* start_dim, const char* start_dimType, void* end_dim, const char* end_dimType, void* out_dim, const char* out_dimType) {}
void lantern_flatten(void* self, const char* selfType, void* dims, const char* dimsType, void* out_dim, const char* out_dimType) {}
void lantern_fill_(void* self, const char* selfType, void* value, const char* valueType) {}
void lantern_fill_(void* self, const char* selfType, void* value, const char* valueType) {}
void lantern_floor(void* self, const char* selfType) {}
void lantern_floor_(void* self, const char* selfType) {}
void lantern_floor_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_frac(void* self, const char* selfType) {}
void lantern_frac_(void* self, const char* selfType) {}
void lantern_frac_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_full(void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_full(void* size, const char* sizeType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType) {}
void lantern_full_out(void* out, const char* outType, void* size, const char* sizeType, void* fill_value, const char* fill_valueType) {}
void lantern_full_like(void* self, const char* selfType, void* fill_value, const char* fill_valueType) {}
void lantern_full_like(void* self, const char* selfType, void* fill_value, const char* fill_valueType, void* options, const char* optionsType) {}
void lantern_from_file(void* filename, const char* filenameType, void* shared, const char* sharedType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_grid_sampler(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType) {}
void lantern_grid_sampler_2d(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType) {}
void lantern_grid_sampler_2d_backward(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType) {}
void lantern_grid_sampler_3d(void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType) {}
void lantern_grid_sampler_3d_backward(void* grad_output, const char* grad_outputType, void* input, const char* inputType, void* grid, const char* gridType, void* interpolation_mode, const char* interpolation_modeType, void* padding_mode, const char* padding_modeType, void* align_corners, const char* align_cornersType) {}
void lantern_hann_window(void* window_length, const char* window_lengthType, void* options, const char* optionsType) {}
void lantern_hann_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType) {}
void lantern_hamming_window(void* window_length, const char* window_lengthType, void* options, const char* optionsType) {}
void lantern_hamming_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* options, const char* optionsType) {}
void lantern_hamming_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* alpha, const char* alphaType, void* options, const char* optionsType) {}
void lantern_hamming_window(void* window_length, const char* window_lengthType, void* periodic, const char* periodicType, void* alpha, const char* alphaType, void* beta, const char* betaType, void* options, const char* optionsType) {}
void lantern_hinge_embedding_loss(void* self, const char* selfType, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType) {}
void lantern_ger(void* self, const char* selfType, void* vec2, const char* vec2Type) {}
void lantern_ger_out(void* out, const char* outType, void* self, const char* selfType, void* vec2, const char* vec2Type) {}
void lantern_group_norm(void* input, const char* inputType, void* num_groups, const char* num_groupsType, void* weight, const char* weightType, void* bias, const char* biasType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType) {}
void lantern_fft(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType) {}
void lantern_ifft(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType) {}
void lantern_rfft(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType) {}
void lantern_irfft(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType, void* signal_sizes, const char* signal_sizesType) {}
void lantern__fft_with_size(void* self, const char* selfType, void* signal_ndim, const char* signal_ndimType, void* complex_input, const char* complex_inputType, void* complex_output, const char* complex_outputType, void* inverse, const char* inverseType, void* checked_signal_sizes, const char* checked_signal_sizesType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType, void* output_sizes, const char* output_sizesType) {}
void lantern__cufft_get_plan_cache_size(void* device_index, const char* device_indexType) {}
void lantern__cufft_get_plan_cache_max_size(void* device_index, const char* device_indexType) {}
void lantern__cufft_set_plan_cache_max_size(void* device_index, const char* device_indexType, void* max_size, const char* max_sizeType) {}
void lantern__cufft_clear_plan_cache(void* device_index, const char* device_indexType) {}
void lantern_index(void* self, const char* selfType, void* indices, const char* indicesType) {}
void lantern_index_copy_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_copy(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_copy_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_copy(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_put_(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType) {}
void lantern_index_put(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType) {}
void lantern__index_put_impl_(void* self, const char* selfType, void* indices, const char* indicesType, void* values, const char* valuesType, void* accumulate, const char* accumulateType, void* unsafe, const char* unsafeType) {}
void lantern_instance_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* use_input_stats, const char* use_input_statsType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* cudnn_enabled, const char* cudnn_enabledType) {}
void lantern_inverse(void* self, const char* selfType) {}
void lantern_inverse_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern__inverse_helper(void* self, const char* selfType) {}
void lantern_isclose(void* self, const char* selfType, void* other, const char* otherType, void* rtol, const char* rtolType, void* atol, const char* atolType, void* equal_nan, const char* equal_nanType) {}
void lantern_isnan(void* self, const char* selfType) {}
void lantern_is_distributed(void* self, const char* selfType) {}
void lantern_is_floating_point(void* self, const char* selfType) {}
void lantern_is_complex(void* self, const char* selfType) {}
void lantern_is_nonzero(void* self, const char* selfType) {}
void lantern_is_same_size(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_is_signed(void* self, const char* selfType) {}
void lantern_kl_div(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_kl_div_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_kthvalue(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_kthvalue_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_kthvalue(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_kthvalue_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_layer_norm(void* input, const char* inputType, void* normalized_shape, const char* normalized_shapeType, void* weight, const char* weightType, void* bias, const char* biasType, void* eps, const char* epsType, void* cudnn_enable, const char* cudnn_enableType) {}
void lantern_native_layer_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* M, const char* MType, void* N, const char* NType, void* eps, const char* epsType) {}
void lantern_native_layer_norm_backward(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* rstd, const char* rstdType, void* weight, const char* weightType, void* M, const char* MType, void* N, const char* NType, void* output_mask, const char* output_maskType) {}
void lantern_native_layer_norm_double_backward(void* ggI, const char* ggIType, void* ggW, const char* ggWType, void* ggb, const char* ggbType, void* gO, const char* gOType, void* input, const char* inputType, void* mean, const char* meanType, void* rstd, const char* rstdType, void* weight, const char* weightType, void* M, const char* MType, void* N, const char* NType, void* output_mask, const char* output_maskType) {}
void lantern_linear(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType) {}
void lantern_mkldnn_linear(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType) {}
void lantern_fbgemm_linear_int8_weight_fp32_activation(void* input, const char* inputType, void* weight, const char* weightType, void* packed, const char* packedType, void* col_offsets, const char* col_offsetsType, void* weight_scale, const char* weight_scaleType, void* weight_zero_point, const char* weight_zero_pointType, void* bias, const char* biasType) {}
void lantern_fbgemm_linear_int8_weight(void* input, const char* inputType, void* weight, const char* weightType, void* packed, const char* packedType, void* col_offsets, const char* col_offsetsType, void* weight_scale, const char* weight_scaleType, void* weight_zero_point, const char* weight_zero_pointType, void* bias, const char* biasType) {}
void lantern_fbgemm_linear_quantize_weight(void* input, const char* inputType) {}
void lantern_fbgemm_pack_gemm_matrix_fp16(void* input, const char* inputType) {}
void lantern_fbgemm_linear_fp16_weight_fp32_activation(void* input, const char* inputType, void* packed_weight, const char* packed_weightType, void* bias, const char* biasType) {}
void lantern_fbgemm_linear_fp16_weight(void* input, const char* inputType, void* packed_weight, const char* packed_weightType, void* bias, const char* biasType) {}
void lantern_fbgemm_pack_quantized_matrix(void* input, const char* inputType) {}
void lantern_fbgemm_pack_quantized_matrix(void* input, const char* inputType, void* K, const char* KType, void* N, const char* NType) {}
void lantern_linspace(void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* options, const char* optionsType) {}
void lantern_linspace_out(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType) {}
void lantern_log(void* self, const char* selfType) {}
void lantern_log_(void* self, const char* selfType) {}
void lantern_log_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_log10(void* self, const char* selfType) {}
void lantern_log10_(void* self, const char* selfType) {}
void lantern_log10_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_log1p(void* self, const char* selfType) {}
void lantern_log1p_(void* self, const char* selfType) {}
void lantern_log1p_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_log2(void* self, const char* selfType) {}
void lantern_log2_(void* self, const char* selfType) {}
void lantern_log2_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_logdet(void* self, const char* selfType) {}
void lantern_logspace(void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* base, const char* baseType, void* options, const char* optionsType) {}
void lantern_logspace_out(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* steps, const char* stepsType, void* base, const char* baseType) {}
void lantern_log_softmax(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_log_softmax(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern__log_softmax(void* self, const char* selfType, void* dim, const char* dimType, void* half_to_float, const char* half_to_floatType) {}
void lantern__log_softmax_backward_data(void* grad_output, const char* grad_outputType, void* output, const char* outputType, void* dim, const char* dimType, void* self, const char* selfType) {}
void lantern_logsumexp(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_logsumexp_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_logsumexp(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_logsumexp_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_margin_ranking_loss(void* input1, const char* input1Type, void* input2, const char* input2Type, void* target, const char* targetType, void* margin, const char* marginType, void* reduction, const char* reductionType) {}
void lantern_matmul(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_matmul_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_matrix_rank(void* self, const char* selfType, void* tol, const char* tolType, void* symmetric, const char* symmetricType) {}
void lantern_matrix_rank(void* self, const char* selfType, void* symmetric, const char* symmetricType) {}
void lantern_matrix_power(void* self, const char* selfType, void* n, const char* nType) {}
void lantern_max(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max_out(void* max, const char* maxType, void* max_values, const char* max_valuesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max_values(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max_out(void* max, const char* maxType, void* max_values, const char* max_valuesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max_values(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_max_pool1d_with_indices(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool1d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool2d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_mkldnn_max_pool2d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_quantized_max_pool2d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool3d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_mean(void* self, const char* selfType, void* dtype, const char* dtypeType) {}
void lantern_mean(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_mean_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_mean(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_mean_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_median(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_median_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_median(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_median_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min_out(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min_values(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min_out(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_min_values(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_mkldnn_convolution(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType) {}
void lantern_mkldnn_convolution_backward_input(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* bias_defined, const char* bias_definedType) {}
void lantern_mkldnn_convolution_backward_weights(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* bias_defined, const char* bias_definedType) {}
void lantern_mkldnn_convolution_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* output_mask, const char* output_maskType) {}
void lantern_miopen_batch_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* exponential_average_factor, const char* exponential_average_factorType, void* epsilon, const char* epsilonType) {}
void lantern_miopen_batch_norm_backward(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_var, const char* save_varType, void* epsilon, const char* epsilonType) {}
void lantern_miopen_convolution(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_convolution_backward_input(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_convolution_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType) {}
void lantern_miopen_convolution_backward_bias(void* grad_output, const char* grad_outputType) {}
void lantern_miopen_convolution_backward_weight(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_convolution_transpose(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_convolution_transpose_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType) {}
void lantern_miopen_convolution_transpose_backward_input(void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_convolution_transpose_backward_weight(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_depthwise_convolution(void* self, const char* selfType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_depthwise_convolution_backward_input(void* self_size, const char* self_sizeType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_depthwise_convolution_backward(void* self, const char* selfType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType, void* output_mask, const char* output_maskType) {}
void lantern_miopen_depthwise_convolution_backward_weight(void* weight_size, const char* weight_sizeType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType, void* benchmark, const char* benchmarkType, void* deterministic, const char* deterministicType) {}
void lantern_miopen_rnn(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* hx, const char* hxType, void* cx, const char* cxType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType) {}
void lantern_miopen_rnn_backward(void* input, const char* inputType, void* weight, const char* weightType, void* weight_stride0, const char* weight_stride0Type, void* weight_buf, const char* weight_bufType, void* hx, const char* hxType, void* cx, const char* cxType, void* output, const char* outputType, void* grad_output, const char* grad_outputType, void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* mode, const char* modeType, void* hidden_size, const char* hidden_sizeType, void* num_layers, const char* num_layersType, void* batch_first, const char* batch_firstType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_sizes, const char* batch_sizesType, void* dropout_state, const char* dropout_stateType, void* reserve, const char* reserveType, void* output_mask, const char* output_maskType) {}
void lantern_mm(void* self, const char* selfType, void* mat2, const char* mat2Type) {}
void lantern_mm_out(void* out, const char* outType, void* self, const char* selfType, void* mat2, const char* mat2Type) {}
void lantern__sparse_mm(void* sparse, const char* sparseType, void* dense, const char* denseType) {}
void lantern_mode(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_mode_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_mode(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_mode_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_mul(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_mul_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_mul_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_mul(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_mul_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_mv(void* self, const char* selfType, void* vec, const char* vecType) {}
void lantern_mv_out(void* out, const char* outType, void* self, const char* selfType, void* vec, const char* vecType) {}
void lantern_mvlgamma(void* self, const char* selfType, void* p, const char* pType) {}
void lantern_mvlgamma_(void* self, const char* selfType, void* p, const char* pType) {}
void lantern_narrow_copy(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* length, const char* lengthType) {}
void lantern_narrow(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* length, const char* lengthType) {}
void lantern_native_batch_norm(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* training, const char* trainingType, void* momentum, const char* momentumType, void* eps, const char* epsType) {}
void lantern_batch_norm_stats(void* input, const char* inputType, void* eps, const char* epsType) {}
void lantern_batch_norm_elemt(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* eps, const char* epsType) {}
void lantern_batch_norm_gather_stats(void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* count, const char* countType) {}
void lantern_batch_norm_gather_stats_with_counts(void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType, void* eps, const char* epsType, void* counts, const char* countsType) {}
void lantern_native_batch_norm_backward(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* weight, const char* weightType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* save_mean, const char* save_meanType, void* save_invstd, const char* save_invstdType, void* train, const char* trainType, void* eps, const char* epsType, void* output_mask, const char* output_maskType) {}
void lantern_batch_norm_backward_reduce(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* weight, const char* weightType, void* input_g, const char* input_gType, void* weight_g, const char* weight_gType, void* bias_g, const char* bias_gType) {}
void lantern_batch_norm_backward_elemt(void* grad_out, const char* grad_outType, void* input, const char* inputType, void* mean, const char* meanType, void* invstd, const char* invstdType, void* weight, const char* weightType, void* mean_dy, const char* mean_dyType, void* mean_dy_xmu, const char* mean_dy_xmuType) {}
void lantern_batch_norm_update_stats(void* input, const char* inputType, void* running_mean, const char* running_meanType, void* running_var, const char* running_varType, void* momentum, const char* momentumType) {}
void lantern__nnpack_available() {}
void lantern__nnpack_spatial_convolution(void* input, const char* inputType, void* weight, const char* weightType, void* bias, const char* biasType, void* padding, const char* paddingType) {}
void lantern__nnpack_spatial_convolution_backward(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType, void* output_mask, const char* output_maskType) {}
void lantern__nnpack_spatial_convolution_backward_input(void* input, const char* inputType, void* grad_output, const char* grad_outputType, void* weight, const char* weightType, void* padding, const char* paddingType) {}
void lantern__nnpack_spatial_convolution_backward_weight(void* input, const char* inputType, void* weightsize, const char* weightsizeType, void* grad_output, const char* grad_outputType, void* padding, const char* paddingType) {}
void lantern_ones(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_ones(void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_ones_out(void* out, const char* outType, void* size, const char* sizeType) {}
void lantern_ones_like(void* self, const char* selfType) {}
void lantern_ones_like(void* self, const char* selfType, void* options, const char* optionsType) {}
void lantern_pairwise_distance(void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType, void* eps, const char* epsType, void* keepdim, const char* keepdimType) {}
void lantern_cdist(void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType) {}
void lantern__cdist_backward(void* grad, const char* gradType, void* x1, const char* x1Type, void* x2, const char* x2Type, void* p, const char* pType, void* cdist, const char* cdistType) {}
void lantern_pdist(void* self, const char* selfType, void* p, const char* pType) {}
void lantern__pdist_forward(void* self, const char* selfType, void* p, const char* pType) {}
void lantern__pdist_backward(void* grad, const char* gradType, void* self, const char* selfType, void* p, const char* pType, void* pdist, const char* pdistType) {}
void lantern_cosine_similarity(void* x1, const char* x1Type, void* x2, const char* x2Type, void* dim, const char* dimType, void* eps, const char* epsType) {}
void lantern_permute(void* self, const char* selfType, void* dims, const char* dimsType) {}
void lantern_numpy_t(void* self, const char* selfType) {}
void lantern_pixel_shuffle(void* self, const char* selfType, void* upscale_factor, const char* upscale_factorType) {}
void lantern_is_pinned(void* self, const char* selfType) {}
void lantern_pin_memory(void* self, const char* selfType) {}
void lantern_pinverse(void* self, const char* selfType, void* rcond, const char* rcondType) {}
void lantern_poisson_nll_loss(void* input, const char* inputType, void* target, const char* targetType, void* log_input, const char* log_inputType, void* full, const char* fullType, void* eps, const char* epsType, void* reduction, const char* reductionType) {}
void lantern_scalar_tensor(void* s, const char* sType, void* options, const char* optionsType) {}
void lantern_rand(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_rand(void* size, const char* sizeType, void* generator, const char* generatorType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_rand(void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_rand(void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_rand_out(void* out, const char* outType, void* size, const char* sizeType) {}
void lantern_rand_out(void* out, const char* outType, void* size, const char* sizeType, void* generator, const char* generatorType) {}
void lantern_rand_like(void* self, const char* selfType) {}
void lantern_rand_like(void* self, const char* selfType, void* options, const char* optionsType) {}
void lantern_randint(void* high, const char* highType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_randint(void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_randint(void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_randint(void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_randint_out(void* out, const char* outType, void* high, const char* highType, void* size, const char* sizeType) {}
void lantern_randint_out(void* out, const char* outType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType) {}
void lantern_randint_out(void* out, const char* outType, void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType) {}
void lantern_randint_out(void* out, const char* outType, void* low, const char* lowType, void* high, const char* highType, void* size, const char* sizeType, void* generator, const char* generatorType) {}
void lantern_randint_like(void* self, const char* selfType, void* high, const char* highType) {}
void lantern_randint_like(void* self, const char* selfType, void* low, const char* lowType, void* high, const char* highType) {}
void lantern_randint_like(void* self, const char* selfType, void* high, const char* highType, void* options, const char* optionsType) {}
void lantern_randint_like(void* self, const char* selfType, void* low, const char* lowType, void* high, const char* highType, void* options, const char* optionsType) {}
void lantern_randn(void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_randn(void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_randn(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_randn(void* size, const char* sizeType, void* generator, const char* generatorType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_randn_out(void* out, const char* outType, void* size, const char* sizeType) {}
void lantern_randn_out(void* out, const char* outType, void* size, const char* sizeType, void* generator, const char* generatorType) {}
void lantern_randn_like(void* self, const char* selfType) {}
void lantern_randn_like(void* self, const char* selfType, void* options, const char* optionsType) {}
void lantern_randperm(void* n, const char* nType, void* options, const char* optionsType) {}
void lantern_randperm(void* n, const char* nType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_randperm_out(void* out, const char* outType, void* n, const char* nType) {}
void lantern_randperm_out(void* out, const char* outType, void* n, const char* nType, void* generator, const char* generatorType) {}
void lantern_range(void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType, void* options, const char* optionsType) {}
void lantern_range(void* start, const char* startType, void* end, const char* endType, void* options, const char* optionsType) {}
void lantern_range_out(void* out, const char* outType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType) {}
void lantern_reciprocal(void* self, const char* selfType) {}
void lantern_reciprocal_(void* self, const char* selfType) {}
void lantern_reciprocal_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_neg(void* self, const char* selfType) {}
void lantern_neg_(void* self, const char* selfType) {}
void lantern_neg_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_repeat(void* self, const char* selfType, void* repeats, const char* repeatsType) {}
void lantern_repeat_interleave(void* repeats, const char* repeatsType) {}
void lantern_repeat_interleave(void* self, const char* selfType, void* repeats, const char* repeatsType, void* dim, const char* dimType) {}
void lantern_repeat_interleave(void* self, const char* selfType, void* repeats, const char* repeatsType, void* dim, const char* dimType) {}
void lantern_reshape(void* self, const char* selfType, void* shape, const char* shapeType) {}
void lantern__mkldnn_reshape(void* self, const char* selfType, void* shape, const char* shapeType) {}
void lantern_reshape_as(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_round(void* self, const char* selfType) {}
void lantern_round_(void* self, const char* selfType) {}
void lantern_round_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_rrelu(void* self, const char* selfType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType) {}
void lantern_rrelu_(void* self, const char* selfType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType) {}
void lantern_relu(void* self, const char* selfType) {}
void lantern_relu_(void* self, const char* selfType) {}
void lantern_prelu(void* self, const char* selfType, void* weight, const char* weightType) {}
void lantern_prelu_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType) {}
void lantern_gelu(void* self, const char* selfType) {}
void lantern_gelu_backward(void* grad, const char* gradType, void* self, const char* selfType) {}
void lantern_hardshrink(void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_hardshrink_backward(void* grad_out, const char* grad_outType, void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_rsqrt(void* self, const char* selfType) {}
void lantern_rsqrt_(void* self, const char* selfType) {}
void lantern_rsqrt_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_select(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_select(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_selu(void* self, const char* selfType) {}
void lantern_selu_(void* self, const char* selfType) {}
void lantern_celu(void* self, const char* selfType, void* alpha, const char* alphaType) {}
void lantern_celu_(void* self, const char* selfType, void* alpha, const char* alphaType) {}
void lantern_sigmoid(void* self, const char* selfType) {}
void lantern_sigmoid_(void* self, const char* selfType) {}
void lantern_sigmoid_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_sin(void* self, const char* selfType) {}
void lantern_sin_(void* self, const char* selfType) {}
void lantern_sin_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_sinh(void* self, const char* selfType) {}
void lantern_sinh_(void* self, const char* selfType) {}
void lantern_sinh_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_detach(void* self, const char* selfType) {}
void lantern_detach_(void* self, const char* selfType) {}
void lantern_size(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_size(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_slice(void* self, const char* selfType, void* dim, const char* dimType, void* start, const char* startType, void* end, const char* endType, void* step, const char* stepType) {}
void lantern_slogdet(void* self, const char* selfType) {}
void lantern_smm(void* self, const char* selfType, void* mat2, const char* mat2Type) {}
void lantern_softmax(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern_softmax(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern__softmax(void* self, const char* selfType, void* dim, const char* dimType, void* half_to_float, const char* half_to_floatType) {}
void lantern__softmax_backward_data(void* grad_output, const char* grad_outputType, void* output, const char* outputType, void* dim, const char* dimType, void* self, const char* selfType) {}
void lantern_split(void* self, const char* selfType, void* split_size, const char* split_sizeType, void* dim, const char* dimType) {}
void lantern_split_with_sizes(void* self, const char* selfType, void* split_sizes, const char* split_sizesType, void* dim, const char* dimType) {}
void lantern_squeeze(void* self, const char* selfType) {}
void lantern_squeeze(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_squeeze(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_squeeze_(void* self, const char* selfType) {}
void lantern_squeeze_(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_squeeze_(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_sspaddmm(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_sspaddmm_out(void* out, const char* outType, void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_stack(void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_stack_out(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern_stft(void* self, const char* selfType, void* n_fft, const char* n_fftType, void* hop_length, const char* hop_lengthType, void* win_length, const char* win_lengthType, void* window, const char* windowType, void* normalized, const char* normalizedType, void* onesided, const char* onesidedType) {}
void lantern_stride(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_stride(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_sum(void* self, const char* selfType, void* dtype, const char* dtypeType) {}
void lantern_sum(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_sum(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_sum_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_sum_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_sum_to_size(void* self, const char* selfType, void* size, const char* sizeType) {}
void lantern_sqrt(void* self, const char* selfType) {}
void lantern_sqrt_(void* self, const char* selfType) {}
void lantern_sqrt_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_std(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern_std(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_std_mean(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern_std_mean(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_std_mean(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_std_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_std(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_std_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_prod(void* self, const char* selfType, void* dtype, const char* dtypeType) {}
void lantern_prod(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_prod_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_prod(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_prod_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_t(void* self, const char* selfType) {}
void lantern_t_(void* self, const char* selfType) {}
void lantern_tan(void* self, const char* selfType) {}
void lantern_tan_(void* self, const char* selfType) {}
void lantern_tan_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_tanh(void* self, const char* selfType) {}
void lantern_tanh_(void* self, const char* selfType) {}
void lantern_tanh_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_tensordot(void* self, const char* selfType, void* other, const char* otherType, void* dims_self, const char* dims_selfType, void* dims_other, const char* dims_otherType) {}
void lantern_threshold(void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType) {}
void lantern_threshold_(void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType) {}
void lantern_threshold_out(void* out, const char* outType, void* self, const char* selfType, void* threshold, const char* thresholdType, void* value, const char* valueType) {}
void lantern_threshold_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* threshold, const char* thresholdType) {}
void lantern_transpose(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type) {}
void lantern_transpose(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type) {}
void lantern__mkldnn_transpose(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type) {}
void lantern_transpose_(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type) {}
void lantern__mkldnn_transpose_(void* self, const char* selfType, void* dim0, const char* dim0Type, void* dim1, const char* dim1Type) {}
void lantern_one_hot(void* self, const char* selfType, void* num_classes, const char* num_classesType) {}
void lantern_flip(void* self, const char* selfType, void* dims, const char* dimsType) {}
void lantern_roll(void* self, const char* selfType, void* shifts, const char* shiftsType, void* dims, const char* dimsType) {}
void lantern_rot90(void* self, const char* selfType, void* k, const char* kType, void* dims, const char* dimsType) {}
void lantern_trapz(void* y, const char* yType, void* x, const char* xType, void* dim, const char* dimType) {}
void lantern_trapz(void* y, const char* yType, void* dx, const char* dxType, void* dim, const char* dimType) {}
void lantern__trilinear(void* i1, const char* i1Type, void* i2, const char* i2Type, void* i3, const char* i3Type, void* expand1, const char* expand1Type, void* expand2, const char* expand2Type, void* expand3, const char* expand3Type, void* sumdim, const char* sumdimType, void* unroll_dim, const char* unroll_dimType) {}
void lantern_triplet_margin_loss(void* anchor, const char* anchorType, void* positive, const char* positiveType, void* negative, const char* negativeType, void* margin, const char* marginType, void* p, const char* pType, void* eps, const char* epsType, void* swap, const char* swapType, void* reduction, const char* reductionType) {}
void lantern_trunc(void* self, const char* selfType) {}
void lantern_trunc_(void* self, const char* selfType) {}
void lantern_trunc_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_type_as(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern__has_compatible_shallow_copy_type(void* self, const char* selfType, void* from, const char* fromType) {}
void lantern__unique(void* self, const char* selfType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType) {}
void lantern_unique_dim(void* self, const char* selfType, void* dim, const char* dimType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType) {}
void lantern_unique_consecutive(void* self, const char* selfType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType, void* dim, const char* dimType) {}
void lantern_unique_dim_consecutive(void* self, const char* selfType, void* dim, const char* dimType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType) {}
void lantern__unique2(void* self, const char* selfType, void* sorted, const char* sortedType, void* return_inverse, const char* return_inverseType, void* return_counts, const char* return_countsType) {}
void lantern__unsafe_view(void* self, const char* selfType, void* size, const char* sizeType) {}
void lantern_unsqueeze(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_unsqueeze_(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_var(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern_var(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_var_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_var(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_var_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_var_mean(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern_var_mean(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_var_mean(void* self, const char* selfType, void* dim, const char* dimType, void* unbiased, const char* unbiasedType, void* keepdim, const char* keepdimType) {}
void lantern_view_as(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_where(void* condition, const char* conditionType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_where(void* condition, const char* conditionType) {}
void lantern__s_where(void* condition, const char* conditionType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_norm_except_dim(void* v, const char* vType, void* pow, const char* powType, void* dim, const char* dimType) {}
void lantern__weight_norm(void* v, const char* vType, void* g, const char* gType, void* dim, const char* dimType) {}
void lantern__weight_norm_cuda_interface(void* v, const char* vType, void* g, const char* gType, void* dim, const char* dimType) {}
void lantern__weight_norm_cuda_interface_backward(void* grad_w, const char* grad_wType, void* saved_v, const char* saved_vType, void* saved_g, const char* saved_gType, void* saved_norms, const char* saved_normsType, void* dim, const char* dimType) {}
void lantern__weight_norm_differentiable_backward(void* grad_w, const char* grad_wType, void* saved_v, const char* saved_vType, void* saved_g, const char* saved_gType, void* saved_norms, const char* saved_normsType, void* dim, const char* dimType) {}
void lantern_zeros(void* size, const char* sizeType, void* names, const char* namesType, void* options, const char* optionsType) {}
void lantern_zeros(void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_zeros_out(void* out, const char* outType, void* size, const char* sizeType) {}
void lantern_zeros_like(void* self, const char* selfType) {}
void lantern_zeros_like(void* self, const char* selfType, void* options, const char* optionsType) {}
void lantern__standard_gamma_grad(void* self, const char* selfType, void* output, const char* outputType) {}
void lantern__standard_gamma(void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern__dirichlet_grad(void* x, const char* xType, void* alpha, const char* alphaType, void* total, const char* totalType) {}
void lantern__sample_dirichlet(void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern_poisson(void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern_native_norm(void* self, const char* selfType, void* p, const char* pType) {}
void lantern__sparse_sum(void* self, const char* selfType) {}
void lantern__sparse_sum(void* self, const char* selfType, void* dtype, const char* dtypeType) {}
void lantern__sparse_sum(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern__sparse_sum(void* self, const char* selfType, void* dim, const char* dimType, void* dtype, const char* dtypeType) {}
void lantern__sparse_sum_backward(void* grad, const char* gradType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType, void* dtype, const char* dtypeType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_norm_out(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_norm_out(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_norm(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_norm_out(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType, void* dtype, const char* dtypeType) {}
void lantern_norm_out(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_frobenius_norm(void* self, const char* selfType) {}
void lantern_frobenius_norm(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_frobenius_norm_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_nuclear_norm(void* self, const char* selfType, void* keepdim, const char* keepdimType) {}
void lantern_nuclear_norm_out(void* out, const char* outType, void* self, const char* selfType, void* keepdim, const char* keepdimType) {}
void lantern_nuclear_norm(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_nuclear_norm_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_clone(void* self, const char* selfType) {}
void lantern_resize_as_(void* self, const char* selfType, void* the_template, const char* the_templateType) {}
void lantern_pow_out(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_pow(void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_zero_(void* self, const char* selfType) {}
void lantern_sub_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_sub(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_sub_(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_sub(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_sub_(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_rsub(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern_rsub(void* self, const char* selfType, void* other, const char* otherType, void* alpha, const char* alphaType) {}
void lantern__sparse_addmm(void* self, const char* selfType, void* sparse, const char* sparseType, void* dense, const char* denseType, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addmm_out(void* out, const char* outType, void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addmm(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addmm_(void* self, const char* selfType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_sparse_coo_tensor(void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern_sparse_coo_tensor(void* indices, const char* indicesType, void* values, const char* valuesType, void* options, const char* optionsType) {}
void lantern_sparse_coo_tensor(void* indices, const char* indicesType, void* values, const char* valuesType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern__sparse_coo_tensor_unsafe(void* indices, const char* indicesType, void* values, const char* valuesType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern__sparse_coo_tensor_with_dims(void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType, void* size, const char* sizeType, void* options, const char* optionsType) {}
void lantern__sparse_coo_tensor_with_dims_and_tensors(void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType, void* size, const char* sizeType, void* indices, const char* indicesType, void* values, const char* valuesType, void* options, const char* optionsType) {}
void lantern_sparse_resize_(void* self, const char* selfType, void* size, const char* sizeType, void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType) {}
void lantern_sparse_resize_and_clear_(void* self, const char* selfType, void* size, const char* sizeType, void* sparse_dim, const char* sparse_dimType, void* dense_dim, const char* dense_dimType) {}
void lantern_sparse_mask(void* self, const char* selfType, void* mask, const char* maskType) {}
void lantern_to_dense(void* self, const char* selfType) {}
void lantern_to_dense_backward(void* grad, const char* gradType, void* input, const char* inputType) {}
void lantern_sparse_dim(void* self, const char* selfType) {}
void lantern__dimi(void* self, const char* selfType) {}
void lantern_dense_dim(void* self, const char* selfType) {}
void lantern__dimv(void* self, const char* selfType) {}
void lantern__nnz(void* self, const char* selfType) {}
void lantern_coalesce(void* self, const char* selfType) {}
void lantern_is_coalesced(void* self, const char* selfType) {}
void lantern__indices(void* self, const char* selfType) {}
void lantern__values(void* self, const char* selfType) {}
void lantern__coalesced_(void* self, const char* selfType, void* coalesced, const char* coalescedType) {}
void lantern_indices(void* self, const char* selfType) {}
void lantern_values(void* self, const char* selfType) {}
void lantern_hspmm_out(void* out, const char* outType, void* mat1, const char* mat1Type, void* mat2, const char* mat2Type) {}
void lantern_hspmm(void* mat1, const char* mat1Type, void* mat2, const char* mat2Type) {}
void lantern_copy_sparse_to_sparse_(void* self, const char* selfType, void* src, const char* srcType, void* non_blocking, const char* non_blockingType) {}
void lantern_numel(void* self, const char* selfType) {}
void lantern_unbind(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_unbind(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_to_sparse(void* self, const char* selfType, void* sparse_dim, const char* sparse_dimType) {}
void lantern_to_sparse(void* self, const char* selfType) {}
void lantern_to_mkldnn(void* self, const char* selfType) {}
void lantern_mkldnn_reorder_conv2d_weight(void* self, const char* selfType, void* padding, const char* paddingType, void* stride, const char* strideType, void* dilation, const char* dilationType, void* groups, const char* groupsType) {}
void lantern_to_mkldnn_backward(void* grad, const char* gradType, void* input, const char* inputType) {}
void lantern_quantize_per_tensor(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* dtype, const char* dtypeType) {}
void lantern_quantize_per_channel(void* self, const char* selfType, void* scales, const char* scalesType, void* zero_points, const char* zero_pointsType, void* axis, const char* axisType, void* dtype, const char* dtypeType) {}
void lantern_dequantize(void* self, const char* selfType) {}
void lantern_q_scale(void* self, const char* selfType) {}
void lantern_q_zero_point(void* self, const char* selfType) {}
void lantern_q_per_channel_scales(void* self, const char* selfType) {}
void lantern_q_per_channel_zero_points(void* self, const char* selfType) {}
void lantern_q_per_channel_axis(void* self, const char* selfType) {}
void lantern_int_repr(void* self, const char* selfType) {}
void lantern__make_per_tensor_quantized_tensor(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType) {}
void lantern__make_per_channel_quantized_tensor(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType) {}
void lantern_qscheme(void* self, const char* selfType) {}
void lantern_fake_quantize_per_tensor_affine(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType) {}
void lantern_fake_quantize_per_tensor_affine_backward(void* grad, const char* gradType, void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType) {}
void lantern_fake_quantize_per_channel_affine(void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType) {}
void lantern_fake_quantize_per_channel_affine_backward(void* grad, const char* gradType, void* self, const char* selfType, void* scale, const char* scaleType, void* zero_point, const char* zero_pointType, void* axis, const char* axisType, void* quant_min, const char* quant_minType, void* quant_max, const char* quant_maxType) {}
void lantern_to(void* self, const char* selfType, void* options, const char* optionsType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType) {}
void lantern_to(void* self, const char* selfType, void* device, const char* deviceType, void* dtype, const char* dtypeType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType) {}
void lantern_to(void* self, const char* selfType, void* dtype, const char* dtypeType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType) {}
void lantern_to(void* self, const char* selfType, void* other, const char* otherType, void* non_blocking, const char* non_blockingType, void* copy, const char* copyType) {}
void lantern_meshgrid(void* tensors, const char* tensorsType) {}
void lantern_cartesian_prod(void* tensors, const char* tensorsType) {}
void lantern_combinations(void* self, const char* selfType, void* r, const char* rType, void* with_replacement, const char* with_replacementType) {}
void lantern_item(void* self, const char* selfType) {}
void lantern_result_type(void* tensor, const char* tensorType, void* other, const char* otherType) {}
void lantern_result_type(void* tensor, const char* tensorType, void* other, const char* otherType) {}
void lantern_result_type(void* scalar, const char* scalarType, void* tensor, const char* tensorType) {}
void lantern_result_type(void* scalar1, const char* scalar1Type, void* scalar2, const char* scalar2Type) {}
void lantern_can_cast(void* from, const char* fromType, void* to, const char* toType) {}
void lantern_promote_types(void* type1, const char* type1Type, void* type2, const char* type2Type) {}
void lantern__local_scalar_dense(void* self, const char* selfType) {}
void lantern__thnn_fused_lstm_cell(void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* cx, const char* cxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType) {}
void lantern__thnn_fused_lstm_cell_backward(void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* cx, const char* cxType, void* cy, const char* cyType, void* workspace, const char* workspaceType, void* has_bias, const char* has_biasType) {}
void lantern__thnn_differentiable_lstm_cell_backward(void* grad_hy, const char* grad_hyType, void* grad_cy, const char* grad_cyType, void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType, void* cx, const char* cxType, void* cy, const char* cyType) {}
void lantern__thnn_fused_gru_cell(void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* hx, const char* hxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType) {}
void lantern__thnn_fused_gru_cell_backward(void* grad_hy, const char* grad_hyType, void* workspace, const char* workspaceType, void* has_bias, const char* has_biasType) {}
void lantern__thnn_differentiable_gru_cell_backward(void* grad_hy, const char* grad_hyType, void* input_gates, const char* input_gatesType, void* hidden_gates, const char* hidden_gatesType, void* hx, const char* hxType, void* input_bias, const char* input_biasType, void* hidden_bias, const char* hidden_biasType) {}
void lantern_lstm(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType) {}
void lantern_lstm(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType) {}
void lantern_gru(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType) {}
void lantern_gru(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType) {}
void lantern_rnn_tanh(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType) {}
void lantern_rnn_tanh(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType) {}
void lantern_rnn_relu(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType) {}
void lantern_rnn_relu(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType) {}
void lantern_lstm_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType) {}
void lantern_gru_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType) {}
void lantern_rnn_tanh_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType) {}
void lantern_rnn_relu_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType) {}
void lantern_quantized_lstm(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType, void* dtype, const char* dtypeType, void* use_dynamic, const char* use_dynamicType) {}
void lantern_quantized_gru(void* input, const char* inputType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType, void* batch_first, const char* batch_firstType) {}
void lantern_quantized_gru(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* hx, const char* hxType, void* params, const char* paramsType, void* has_biases, const char* has_biasesType, void* num_layers, const char* num_layersType, void* dropout, const char* dropoutType, void* train, const char* trainType, void* bidirectional, const char* bidirectionalType) {}
void lantern_quantized_lstm_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType) {}
void lantern_quantized_gru_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType) {}
void lantern_quantized_rnn_relu_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType) {}
void lantern_quantized_rnn_tanh_cell(void* input, const char* inputType, void* hx, const char* hxType, void* w_ih, const char* w_ihType, void* w_hh, const char* w_hhType, void* b_ih, const char* b_ihType, void* b_hh, const char* b_hhType, void* packed_ih, const char* packed_ihType, void* packed_hh, const char* packed_hhType, void* col_offsets_ih, const char* col_offsets_ihType, void* col_offsets_hh, const char* col_offsets_hhType, void* scale_ih, const char* scale_ihType, void* scale_hh, const char* scale_hhType, void* zero_point_ih, const char* zero_point_ihType, void* zero_point_hh, const char* zero_point_hhType) {}
void lantern__pack_padded_sequence(void* input, const char* inputType, void* lengths, const char* lengthsType, void* batch_first, const char* batch_firstType) {}
void lantern__pack_padded_sequence_backward(void* grad, const char* gradType, void* input_size, const char* input_sizeType, void* batch_sizes, const char* batch_sizesType, void* batch_first, const char* batch_firstType) {}
void lantern__pad_packed_sequence(void* data, const char* dataType, void* batch_sizes, const char* batch_sizesType, void* batch_first, const char* batch_firstType, void* padding_value, const char* padding_valueType, void* total_length, const char* total_lengthType) {}
void lantern_set_(void* self, const char* selfType, void* source, const char* sourceType) {}
void lantern_set_(void* self, const char* selfType, void* source, const char* sourceType, void* storage_offset, const char* storage_offsetType, void* size, const char* sizeType, void* stride, const char* strideType) {}
void lantern_set_(void* self, const char* selfType, void* source, const char* sourceType) {}
void lantern_set_(void* self, const char* selfType) {}
void lantern_set_quantizer_(void* self, const char* selfType, void* quantizer, const char* quantizerType) {}
void lantern_is_set_to(void* self, const char* selfType, void* tensor, const char* tensorType) {}
void lantern_masked_fill_(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType) {}
void lantern_masked_fill(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType) {}
void lantern_masked_fill_(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType) {}
void lantern_masked_fill(void* self, const char* selfType, void* mask, const char* maskType, void* value, const char* valueType) {}
void lantern_masked_scatter_(void* self, const char* selfType, void* mask, const char* maskType, void* source, const char* sourceType) {}
void lantern_masked_scatter(void* self, const char* selfType, void* mask, const char* maskType, void* source, const char* sourceType) {}
void lantern_view(void* self, const char* selfType, void* size, const char* sizeType) {}
void lantern_put_(void* self, const char* selfType, void* index, const char* indexType, void* source, const char* sourceType, void* accumulate, const char* accumulateType) {}
void lantern_index_add_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_add(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_add(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern_index_fill_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_index_fill(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_scatter_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_scatter(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_scatter_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_scatter(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_scatter(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_scatter(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* value, const char* valueType) {}
void lantern_scatter_add_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_scatter_add(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_scatter_add(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* src, const char* srcType) {}
void lantern_lt_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lt_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ne_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ne_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___and__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___and__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___iand__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___iand__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___or__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___or__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ior__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ior__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___xor__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___xor__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ixor__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ixor__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___lshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___lshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ilshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___ilshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___rshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___rshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___irshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern___irshift__(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lgamma_(void* self, const char* selfType) {}
void lantern_atan2_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_tril_(void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_triu_(void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_digamma_(void* self, const char* selfType) {}
void lantern_polygamma_(void* self, const char* selfType, void* n, const char* nType) {}
void lantern_renorm_(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType) {}
void lantern_pow_(void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_pow_(void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_lerp_(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_lerp_(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_fmod_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_fmod_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder_(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_addbmm_(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addbmm_out(void* out, const char* outType, void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addbmm(void* self, const char* selfType, void* batch1, const char* batch1Type, void* batch2, const char* batch2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern_addcdiv_(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_random_(void* self, const char* selfType, void* from, const char* fromType, void* to, const char* toType, void* generator, const char* generatorType) {}
void lantern_random_(void* self, const char* selfType, void* to, const char* toType, void* generator, const char* generatorType) {}
void lantern_random_(void* self, const char* selfType, void* generator, const char* generatorType) {}
void lantern_uniform_(void* self, const char* selfType, void* from, const char* fromType, void* to, const char* toType, void* generator, const char* generatorType) {}
void lantern_normal_(void* self, const char* selfType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_cauchy_(void* self, const char* selfType, void* median, const char* medianType, void* sigma, const char* sigmaType, void* generator, const char* generatorType) {}
void lantern_log_normal_(void* self, const char* selfType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_exponential_(void* self, const char* selfType, void* lambd, const char* lambdType, void* generator, const char* generatorType) {}
void lantern_geometric_(void* self, const char* selfType, void* p, const char* pType, void* generator, const char* generatorType) {}
void lantern_diag_out(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_diag(void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_cross_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType, void* dim, const char* dimType) {}
void lantern_cross(void* self, const char* selfType, void* other, const char* otherType, void* dim, const char* dimType) {}
void lantern_triu_out(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_triu(void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_tril_out(void* out, const char* outType, void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_tril(void* self, const char* selfType, void* diagonal, const char* diagonalType) {}
void lantern_tril_indices(void* row, const char* rowType, void* col, const char* colType, void* offset, const char* offsetType, void* options, const char* optionsType) {}
void lantern_triu_indices(void* row, const char* rowType, void* col, const char* colType, void* offset, const char* offsetType, void* options, const char* optionsType) {}
void lantern_trace(void* self, const char* selfType) {}
void lantern_ne_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ne(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ne_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ne(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_eq(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_ge(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_le(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_gt(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lt_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lt(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lt_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lt(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_take_out(void* out, const char* outType, void* self, const char* selfType, void* index, const char* indexType) {}
void lantern_take(void* self, const char* selfType, void* index, const char* indexType) {}
void lantern_index_select_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_index_select(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_index_select_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_index_select(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType) {}
void lantern_masked_select_out(void* out, const char* outType, void* self, const char* selfType, void* mask, const char* maskType) {}
void lantern_masked_select(void* self, const char* selfType, void* mask, const char* maskType) {}
void lantern_nonzero_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_nonzero(void* self, const char* selfType) {}
void lantern_nonzero_numpy(void* self, const char* selfType) {}
void lantern_gather_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType) {}
void lantern_gather(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType) {}
void lantern_gather_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType) {}
void lantern_gather(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* sparse_grad, const char* sparse_gradType) {}
void lantern__gather_sparse_backward(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* grad, const char* gradType) {}
void lantern_addcmul_out(void* out, const char* outType, void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_addcmul(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_addcmul_(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_addcdiv_out(void* out, const char* outType, void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_addcdiv(void* self, const char* selfType, void* tensor1, const char* tensor1Type, void* tensor2, const char* tensor2Type, void* value, const char* valueType) {}
void lantern_lstsq_out(void* X, const char* XType, void* qr, const char* qrType, void* self, const char* selfType, void* A, const char* AType) {}
void lantern_lstsq(void* self, const char* selfType, void* A, const char* AType) {}
void lantern_triangular_solve_out(void* X, const char* XType, void* M, const char* MType, void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType) {}
void lantern_triangular_solve(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType) {}
void lantern__triangular_solve_helper(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType, void* transpose, const char* transposeType, void* unitriangular, const char* unitriangularType) {}
void lantern_symeig_out(void* e, const char* eType, void* V, const char* VType, void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType) {}
void lantern_symeig(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType) {}
void lantern__symeig_helper(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType, void* upper, const char* upperType) {}
void lantern_eig_out(void* e, const char* eType, void* v, const char* vType, void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType) {}
void lantern_eig(void* self, const char* selfType, void* eigenvectors, const char* eigenvectorsType) {}
void lantern_svd_out(void* U, const char* UType, void* S, const char* SType, void* V, const char* VType, void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType) {}
void lantern_svd(void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType) {}
void lantern__svd_helper(void* self, const char* selfType, void* some, const char* someType, void* compute_uv, const char* compute_uvType) {}
void lantern_cholesky_out(void* out, const char* outType, void* self, const char* selfType, void* upper, const char* upperType) {}
void lantern_cholesky(void* self, const char* selfType, void* upper, const char* upperType) {}
void lantern__cholesky_helper(void* self, const char* selfType, void* upper, const char* upperType) {}
void lantern_cholesky_solve_out(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type, void* upper, const char* upperType) {}
void lantern_cholesky_solve(void* self, const char* selfType, void* input2, const char* input2Type, void* upper, const char* upperType) {}
void lantern__cholesky_solve_helper(void* self, const char* selfType, void* A, const char* AType, void* upper, const char* upperType) {}
void lantern_solve(void* self, const char* selfType, void* A, const char* AType) {}
void lantern_solve_out(void* solution, const char* solutionType, void* lu, const char* luType, void* self, const char* selfType, void* A, const char* AType) {}
void lantern__solve_helper(void* self, const char* selfType, void* A, const char* AType) {}
void lantern_cholesky_inverse_out(void* out, const char* outType, void* self, const char* selfType, void* upper, const char* upperType) {}
void lantern_cholesky_inverse(void* self, const char* selfType, void* upper, const char* upperType) {}
void lantern_qr_out(void* Q, const char* QType, void* R, const char* RType, void* self, const char* selfType, void* some, const char* someType) {}
void lantern_qr(void* self, const char* selfType, void* some, const char* someType) {}
void lantern__qr_helper(void* self, const char* selfType, void* some, const char* someType) {}
void lantern_geqrf_out(void* a, const char* aType, void* tau, const char* tauType, void* self, const char* selfType) {}
void lantern_geqrf(void* self, const char* selfType) {}
void lantern_orgqr_out(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type) {}
void lantern_orgqr(void* self, const char* selfType, void* input2, const char* input2Type) {}
void lantern_ormqr_out(void* out, const char* outType, void* self, const char* selfType, void* input2, const char* input2Type, void* input3, const char* input3Type, void* left, const char* leftType, void* transpose, const char* transposeType) {}
void lantern_ormqr(void* self, const char* selfType, void* input2, const char* input2Type, void* input3, const char* input3Type, void* left, const char* leftType, void* transpose, const char* transposeType) {}
void lantern__lu_with_info(void* self, const char* selfType, void* pivot, const char* pivotType, void* check_errors, const char* check_errorsType) {}
void lantern_lu_solve_out(void* out, const char* outType, void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType) {}
void lantern_lu_solve(void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType) {}
void lantern__lu_solve_helper(void* self, const char* selfType, void* LU_data, const char* LU_dataType, void* LU_pivots, const char* LU_pivotsType) {}
void lantern_multinomial_out(void* out, const char* outType, void* self, const char* selfType, void* num_samples, const char* num_samplesType, void* replacement, const char* replacementType, void* generator, const char* generatorType) {}
void lantern_multinomial(void* self, const char* selfType, void* num_samples, const char* num_samplesType, void* replacement, const char* replacementType, void* generator, const char* generatorType) {}
void lantern__multinomial_alias_setup(void* probs, const char* probsType) {}
void lantern__multinomial_alias_draw(void* J, const char* JType, void* q, const char* qType, void* num_samples, const char* num_samplesType, void* generator, const char* generatorType) {}
void lantern_lgamma_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_lgamma(void* self, const char* selfType) {}
void lantern_digamma_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_digamma(void* self, const char* selfType) {}
void lantern_polygamma_out(void* out, const char* outType, void* n, const char* nType, void* self, const char* selfType) {}
void lantern_polygamma(void* n, const char* nType, void* self, const char* selfType) {}
void lantern_erfinv(void* self, const char* selfType) {}
void lantern_erfinv_(void* self, const char* selfType) {}
void lantern_erfinv_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_sign(void* self, const char* selfType) {}
void lantern_sign_(void* self, const char* selfType) {}
void lantern_sign_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_dist(void* self, const char* selfType, void* other, const char* otherType, void* p, const char* pType) {}
void lantern_atan2_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_atan2(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_lerp_out(void* out, const char* outType, void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_lerp_out(void* out, const char* outType, void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_lerp(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_lerp(void* self, const char* selfType, void* end, const char* endType, void* weight, const char* weightType) {}
void lantern_histc_out(void* out, const char* outType, void* self, const char* selfType, void* bins, const char* binsType, void* min, const char* minType, void* max, const char* maxType) {}
void lantern_histc(void* self, const char* selfType, void* bins, const char* binsType, void* min, const char* minType, void* max, const char* maxType) {}
void lantern_fmod_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_fmod(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_fmod_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_fmod(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_remainder(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_min_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_min(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_min(void* self, const char* selfType) {}
void lantern_max_out(void* out, const char* outType, void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_max(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_max(void* self, const char* selfType) {}
void lantern_median(void* self, const char* selfType) {}
void lantern_sort_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_sort(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_sort_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_sort(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_argsort(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_argsort(void* self, const char* selfType, void* dim, const char* dimType, void* descending, const char* descendingType) {}
void lantern_topk_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* largest, const char* largestType, void* sorted, const char* sortedType) {}
void lantern_topk(void* self, const char* selfType, void* k, const char* kType, void* dim, const char* dimType, void* largest, const char* largestType, void* sorted, const char* sortedType) {}
void lantern_all(void* self, const char* selfType) {}
void lantern_any(void* self, const char* selfType) {}
void lantern_renorm_out(void* out, const char* outType, void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType) {}
void lantern_renorm(void* self, const char* selfType, void* p, const char* pType, void* dim, const char* dimType, void* maxnorm, const char* maxnormType) {}
void lantern_unfold(void* self, const char* selfType, void* dimension, const char* dimensionType, void* size, const char* sizeType, void* step, const char* stepType) {}
void lantern_equal(void* self, const char* selfType, void* other, const char* otherType) {}
void lantern_pow_out(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_pow(void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_pow_out(void* out, const char* outType, void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_pow(void* self, const char* selfType, void* exponent, const char* exponentType) {}
void lantern_normal_out(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal_out(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal_out(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal(void* mean, const char* meanType, void* std, const char* stdType, void* generator, const char* generatorType) {}
void lantern_normal(void* mean, const char* meanType, void* std, const char* stdType, void* size, const char* sizeType, void* generator, const char* generatorType, void* options, const char* optionsType) {}
void lantern_normal_out(void* out, const char* outType, void* mean, const char* meanType, void* std, const char* stdType, void* size, const char* sizeType, void* generator, const char* generatorType) {}
void lantern_alias(void* self, const char* selfType) {}
void lantern__addr(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern__addr_(void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern__addr_out(void* out, const char* outType, void* self, const char* selfType, void* vec1, const char* vec1Type, void* vec2, const char* vec2Type, void* beta, const char* betaType, void* alpha, const char* alphaType) {}
void lantern__index_copy_(void* self, const char* selfType, void* dim, const char* dimType, void* index, const char* indexType, void* source, const char* sourceType) {}
void lantern__cumsum(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern__cumsum_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern__cumprod(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern__cumprod_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern__var(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern__std(void* self, const char* selfType, void* unbiased, const char* unbiasedType) {}
void lantern__cat(void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern__cat_out(void* out, const char* outType, void* tensors, const char* tensorsType, void* dim, const char* dimType) {}
void lantern__mode(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern__mode_out(void* values, const char* valuesType, void* indices, const char* indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern__max(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern__max_out(void* max, const char* maxType, void* max_indices, const char* max_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern__min(void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern__min_out(void* min, const char* minType, void* min_indices, const char* min_indicesType, void* self, const char* selfType, void* dim, const char* dimType, void* keepdim, const char* keepdimType) {}
void lantern_binary_cross_entropy_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_binary_cross_entropy(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_binary_cross_entropy_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_binary_cross_entropy_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_mse_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_mse_loss(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_mse_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_mse_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_l1_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_l1_loss(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_l1_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_l1_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_multi_margin_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_multi_margin_loss(void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_multi_margin_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_multi_margin_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* p, const char* pType, void* margin, const char* marginType, void* weight, const char* weightType, void* reduction, const char* reductionType) {}
void lantern_multilabel_margin_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_multilabel_margin_loss(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_multilabel_margin_loss_forward_out(void* output, const char* outputType, void* is_target, const char* is_targetType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_multilabel_margin_loss_forward(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_multilabel_margin_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType, void* is_target, const char* is_targetType) {}
void lantern_multilabel_margin_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType, void* is_target, const char* is_targetType) {}
void lantern_nll_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss_forward_out(void* output, const char* outputType, void* total_weight, const char* total_weightType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss_forward(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType) {}
void lantern_nll_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType) {}
void lantern_nll_loss2d_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss2d(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss2d_forward_out(void* output, const char* outputType, void* total_weight, const char* total_weightType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss2d_forward(void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType) {}
void lantern_nll_loss2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType) {}
void lantern_nll_loss2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* weight, const char* weightType, void* reduction, const char* reductionType, void* ignore_index, const char* ignore_indexType, void* total_weight, const char* total_weightType) {}
void lantern_smooth_l1_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_smooth_l1_loss(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_smooth_l1_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_smooth_l1_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_soft_margin_loss_out(void* out, const char* outType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_soft_margin_loss(void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_soft_margin_loss_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_soft_margin_loss_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* target, const char* targetType, void* reduction, const char* reductionType) {}
void lantern_elu_out(void* out, const char* outType, void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType) {}
void lantern_elu(void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType) {}
void lantern_elu_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType, void* output, const char* outputType) {}
void lantern_elu_backward(void* grad_output, const char* grad_outputType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType, void* output, const char* outputType) {}
void lantern_elu_(void* self, const char* selfType, void* alpha, const char* alphaType, void* scale, const char* scaleType, void* input_scale, const char* input_scaleType) {}
void lantern_glu_out(void* out, const char* outType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_glu(void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_glu_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_glu_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* dim, const char* dimType) {}
void lantern_hardtanh_out(void* out, const char* outType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType) {}
void lantern_hardtanh(void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType) {}
void lantern_hardtanh_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType) {}
void lantern_hardtanh_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType) {}
void lantern_hardtanh_(void* self, const char* selfType, void* min_val, const char* min_valType, void* max_val, const char* max_valType) {}
void lantern_leaky_relu_out(void* out, const char* outType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType) {}
void lantern_leaky_relu(void* self, const char* selfType, void* negative_slope, const char* negative_slopeType) {}
void lantern_leaky_relu_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType) {}
void lantern_leaky_relu_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* negative_slope, const char* negative_slopeType) {}
void lantern_leaky_relu_(void* self, const char* selfType, void* negative_slope, const char* negative_slopeType) {}
void lantern_log_sigmoid_out(void* out, const char* outType, void* self, const char* selfType) {}
void lantern_log_sigmoid(void* self, const char* selfType) {}
void lantern_log_sigmoid_forward_out(void* output, const char* outputType, void* buffer, const char* bufferType, void* self, const char* selfType) {}
void lantern_log_sigmoid_forward(void* self, const char* selfType) {}
void lantern_log_sigmoid_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* buffer, const char* bufferType) {}
void lantern_log_sigmoid_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* buffer, const char* bufferType) {}
void lantern_rrelu_with_noise_out(void* out, const char* outType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType) {}
void lantern_rrelu_with_noise(void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType) {}
void lantern_rrelu_with_noise_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType) {}
void lantern_rrelu_with_noise_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType) {}
void lantern_rrelu_with_noise_(void* self, const char* selfType, void* noise, const char* noiseType, void* lower, const char* lowerType, void* upper, const char* upperType, void* training, const char* trainingType, void* generator, const char* generatorType) {}
void lantern_softplus_out(void* out, const char* outType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType) {}
void lantern_softplus(void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType) {}
void lantern_softplus_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType, void* output, const char* outputType) {}
void lantern_softplus_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* beta, const char* betaType, void* threshold, const char* thresholdType, void* output, const char* outputType) {}
void lantern_softshrink_out(void* out, const char* outType, void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_softshrink(void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_softshrink_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_softshrink_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* lambd, const char* lambdType) {}
void lantern_adaptive_avg_pool2d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_avg_pool2d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_mkldnn_adaptive_avg_pool2d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern__adaptive_avg_pool2d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern__adaptive_avg_pool2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType) {}
void lantern_adaptive_avg_pool3d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_avg_pool3d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_avg_pool3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType) {}
void lantern_adaptive_avg_pool3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType) {}
void lantern_adaptive_max_pool2d_out(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_max_pool2d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_max_pool2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType) {}
void lantern_adaptive_max_pool2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType) {}
void lantern_adaptive_max_pool3d_out(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_max_pool3d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_adaptive_max_pool3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType) {}
void lantern_adaptive_max_pool3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType) {}
void lantern_avg_pool2d_out(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool2d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool3d_out(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool3d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_avg_pool3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* ceil_mode, const char* ceil_modeType, void* count_include_pad, const char* count_include_padType, void* divisor_override, const char* divisor_overrideType) {}
void lantern_fractional_max_pool2d_out(void* output, const char* outputType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType) {}
void lantern_fractional_max_pool2d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType) {}
void lantern_fractional_max_pool2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType) {}
void lantern_fractional_max_pool2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType) {}
void lantern_fractional_max_pool3d_out(void* output, const char* outputType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType) {}
void lantern_fractional_max_pool3d(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* random_samples, const char* random_samplesType) {}
void lantern_fractional_max_pool3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType) {}
void lantern_fractional_max_pool3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* output_size, const char* output_sizeType, void* indices, const char* indicesType) {}
void lantern_max_pool2d_with_indices_out(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool2d_with_indices(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool2d_with_indices_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType) {}
void lantern_max_pool2d_with_indices_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType) {}
void lantern_max_pool3d_with_indices_out(void* out, const char* outType, void* indices, const char* indicesType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool3d_with_indices(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType) {}
void lantern_max_pool3d_with_indices_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType) {}
void lantern_max_pool3d_with_indices_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* ceil_mode, const char* ceil_modeType, void* indices, const char* indicesType) {}
void lantern_max_unpool2d_out(void* out, const char* outType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType) {}
void lantern_max_unpool2d(void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType) {}
void lantern_max_unpool2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType) {}
void lantern_max_unpool2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType) {}
void lantern_max_unpool3d_out(void* out, const char* outType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_max_unpool3d(void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_max_unpool3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_max_unpool3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* indices, const char* indicesType, void* output_size, const char* output_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_reflection_pad1d_out(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad1d(void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad1d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad1d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad2d_out(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad2d(void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_reflection_pad2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad1d_out(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad1d(void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad1d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad1d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad2d_out(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad2d(void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad3d_out(void* out, const char* outType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad3d(void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_replication_pad3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* padding, const char* paddingType) {}
void lantern_upsample_linear1d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_linear1d(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_linear1d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_linear1d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bilinear2d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bilinear2d(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bilinear2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bilinear2d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bicubic2d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bicubic2d(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bicubic2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_bicubic2d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_trilinear3d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_trilinear3d(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_trilinear3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_trilinear3d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType, void* align_corners, const char* align_cornersType) {}
void lantern_upsample_nearest1d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest1d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest1d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_upsample_nearest1d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_upsample_nearest2d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest2d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_upsample_nearest2d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_upsample_nearest3d_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest3d(void* self, const char* selfType, void* output_size, const char* output_sizeType) {}
void lantern_upsample_nearest3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_upsample_nearest3d_backward(void* grad_output, const char* grad_outputType, void* output_size, const char* output_sizeType, void* input_size, const char* input_sizeType) {}
void lantern_sigmoid_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output, const char* outputType) {}
void lantern_sigmoid_backward(void* grad_output, const char* grad_outputType, void* output, const char* outputType) {}
void lantern_tanh_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* output, const char* outputType) {}
void lantern_tanh_backward(void* grad_output, const char* grad_outputType, void* output, const char* outputType) {}
void lantern_slow_conv_transpose2d_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_transpose2d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_transpose2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* columns, const char* columnsType, void* ones, const char* onesType) {}
void lantern_slow_conv_transpose2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* columns, const char* columnsType, void* ones, const char* onesType, void* output_mask, const char* output_maskType) {}
void lantern_slow_conv_transpose3d_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_transpose3d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_transpose3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType) {}
void lantern_slow_conv_transpose3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* output_padding, const char* output_paddingType, void* dilation, const char* dilationType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType) {}
void lantern_thnn_conv2d_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv2d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv2d_forward_out(void* output, const char* outputType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv2d_forward(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType) {}
void lantern_thnn_conv2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType) {}
void lantern_thnn_conv_depthwise2d_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_thnn_conv_depthwise2d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_thnn_conv_depthwise2d_forward_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_thnn_conv_depthwise2d_forward(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_thnn_conv_depthwise2d_backward_out(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_thnn_conv_depthwise2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType) {}
void lantern_thnn_conv3d_out(void* out, const char* outType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv3d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv3d_forward_out(void* output, const char* outputType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv3d_forward(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType) {}
void lantern_thnn_conv3d_backward_out(void* grad_input, const char* grad_inputType, void* grad_weight, const char* grad_weightType, void* grad_bias, const char* grad_biasType, void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType) {}
void lantern_thnn_conv3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* finput, const char* finputType, void* fgrad_input, const char* fgrad_inputType, void* output_mask, const char* output_maskType) {}
void lantern_slow_conv_dilated2d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_dilated2d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType) {}
void lantern_slow_conv_dilated3d(void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* bias, const char* biasType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType) {}
void lantern_slow_conv_dilated3d_backward(void* grad_output, const char* grad_outputType, void* self, const char* selfType, void* weight, const char* weightType, void* kernel_size, const char* kernel_sizeType, void* stride, const char* strideType, void* padding, const char* paddingType, void* dilation, const char* dilationType, void* output_mask, const char* output_maskType) {}
void lantern_col2im_out(void* out, const char* outType, void* self, const char* selfType, void* output_size, const char* output_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_col2im(void* self, const char* selfType, void* output_size, const char* output_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_col2im_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_col2im_backward(void* grad_output, const char* grad_outputType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_im2col_out(void* out, const char* outType, void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_im2col(void* self, const char* selfType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_im2col_backward_out(void* grad_input, const char* grad_inputType, void* grad_output, const char* grad_outputType, void* input_size, const char* input_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
void lantern_im2col_backward(void* grad_output, const char* grad_outputType, void* input_size, const char* input_sizeType, void* kernel_size, const char* kernel_sizeType, void* dilation, const char* dilationType, void* padding, const char* paddingType, void* stride, const char* strideType) {}
*/
/* Autogen Body -- End */
