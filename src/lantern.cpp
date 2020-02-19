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

template<class T>
class LanternObject
{
private:
    T _object;
public:
    LanternObject(T object)
    {
        _object = object;
    }

    T& get()
    {
        return _object;
    }
};

/* Autogen Body -- Start */
void* lantern__cast_byte_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Byte(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_char_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Char(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_double_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Double(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_float_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Float(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_int_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Int(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_long_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Long(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_short_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Short(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern__cast_half_tensor_bool(void* self, void* non_blocking)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Half(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
}

void* lantern_align_tensors_tensorlist(void* tensors)
{
    return (void *) new LanternObject<torch::TensorList>(torch::align_tensors(
        ((LanternObject<torch::TensorList>*)tensors)->get()));
}

void* lantern__debug_has_internal_overlap_tensor(void* self)
{
    return (void *) new LanternObject<int64_t>(torch::_debug_has_internal_overlap(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(void* self, void* n, void* sobolstate, void* dimension, void* num_generated)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_ff_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<const torch::Tensor &>*)sobolstate)->get(), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get()));
}

void* lantern__sobol_engine_scramble__tensor_tensor_intt(void* self, void* ltm, void* dimension)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_scramble_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)ltm)->get(), ((LanternObject<int64_t>*)dimension)->get()));
}

void* lantern__sobol_engine_initialize_state__tensor_intt(void* self, void* dimension)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_initialize_state_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dimension)->get()));
}

void* lantern__reshape_from_tensor_tensor_tensor(void* self, void* shape)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_reshape_from_tensor(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)shape)->get()));
}

void* lantern__shape_as_tensor_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_shape_as_tensor(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_dropout_tensor_double_bool(void* input, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::dropout(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_dropout__tensor_double_bool(void* self, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::dropout_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_feature_dropout_tensor_double_bool(void* input, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::feature_dropout(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_feature_dropout__tensor_double_bool(void* self, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::feature_dropout_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::alpha_dropout(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::alpha_dropout_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_feature_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::feature_alpha_dropout(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_feature_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
    return (void *) new LanternObject<torch::Tensor>(torch::feature_alpha_dropout_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
}

void* lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad)
{
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool1d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get()));
}

void* lantern_adaptive_avg_pool1d_tensor_intarrayref(void* self, void* output_size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool1d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)output_size)->get()));
}

void* lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::addr_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)vec1)->get(), ((LanternObject<const torch::Tensor &>*)vec2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_affine_grid_generator_tensor_intarrayref_bool(void* theta, void* size, void* align_corners)
{
    return (void *) new LanternObject<torch::Tensor>(torch::affine_grid_generator(
        ((LanternObject<const torch::Tensor &>*)theta)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
}

void* lantern_affine_grid_generator_backward_tensor_intarrayref_bool(void* grad, void* size, void* align_corners)
{
    return (void *) new LanternObject<torch::Tensor>(torch::affine_grid_generator_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
}

void* lantern_all_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::all_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_all_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::all_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_any_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::any_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_any_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::any_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_arange_scalar_tensoroptions(void* end, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_arange_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_arange_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_arange_out_tensor_scalar(void* out, void* end)
{
    return (void *) new LanternObject<torch::Tensor>(torch::arange_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::Scalar>*)end)->get()));
}

void* lantern__dim_arange_tensor_intt(void* like, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_dim_arange(
        ((LanternObject<const torch::Tensor &>*)like)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_baddbmm_mkl_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)batch1)->get(), ((LanternObject<const torch::Tensor &>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_bartlett_window_intt_tensoroptions(void* window_length, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_bartlett_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
    return (void *) new LanternObject<torch::Tensor>(torch::batch_norm(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<const torch::Tensor &>*)running_mean)->get(), ((LanternObject<const torch::Tensor &>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
}

void* lantern_bernoulli_out_tensor_tensor_generator(void* out, void* self, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::bernoulli_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_bilinear_tensor_tensor_tensor_tensor(void* input1, void* input2, void* weight, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::bilinear(
        ((LanternObject<const torch::Tensor &>*)input1)->get(), ((LanternObject<const torch::Tensor &>*)input2)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_with_logits(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)pos_weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_with_logits_backward(
        ((LanternObject<const torch::Tensor &>*)grad_output)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)pos_weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_blackman_window_intt_tensoroptions(void* window_length, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_blackman_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_broadcast_tensors_tensorlist(void* tensors)
{
    return (void *) new LanternObject<torch::TensorList>(torch::broadcast_tensors(
        ((LanternObject<torch::TensorList>*)tensors)->get()));
}

void* lantern_cat_tensorlist_intt(void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cat(
        ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cat_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_cat_tensorlist_dimname(void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cat(
        ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
}

void* lantern_cat_out_tensor_tensorlist_dimname(void* out, void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cat_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
}

void* lantern_chain_matmul_tensorlist(void* matrices)
{
    return (void *) new LanternObject<torch::Tensor>(torch::chain_matmul(
        ((LanternObject<torch::TensorList>*)matrices)->get()));
}

void* lantern_cudnn_is_acceptable_tensor(void* self)
{
    return (void *) new LanternObject<bool>(torch::cudnn_is_acceptable(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_constant_pad_nd_tensor_intarrayref_scalar(void* self, void* pad, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::constant_pad_nd(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)pad)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::convolution(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::convolution_overrideable(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_convolution(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
}

void* lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_convolution_nogroup(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get()));
}

void* lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv1d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv2d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv3d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern_conv_tbc_tensor_tensor_tensor_intt(void* self, void* weight, void* bias, void* pad)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv_tbc(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<int64_t>*)pad)->get()));
}

void* lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose1d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get()));
}

void* lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose2d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get()));
}

void* lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose3d(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get()));
}

void* lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cosine_embedding_loss(
        ((LanternObject<const torch::Tensor &>*)input1)->get(), ((LanternObject<const torch::Tensor &>*)input2)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_cumsum_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_cumsum_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_cumprod_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_cumprod_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ctc_loss(
        ((LanternObject<const torch::Tensor &>*)log_probs)->get(), ((LanternObject<const torch::Tensor &>*)targets)->get(), ((LanternObject<torch::IntArrayRef>*)input_lengths)->get(), ((LanternObject<torch::IntArrayRef>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
}

void* lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ctc_loss(
        ((LanternObject<const torch::Tensor &>*)log_probs)->get(), ((LanternObject<const torch::Tensor &>*)targets)->get(), ((LanternObject<const torch::Tensor &>*)input_lengths)->get(), ((LanternObject<const torch::Tensor &>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
}

void* lantern_dot_out_tensor_tensor_tensor(void* out, void* self, void* tensor)
{
    return (void *) new LanternObject<torch::Tensor>(torch::dot_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)tensor)->get()));
}

void* lantern_einsum_stdstring_tensorlist(void* equation, void* tensors)
{
    return (void *) new LanternObject<torch::Tensor>(torch::einsum(
        ((LanternObject<std::string>*)equation)->get(), ((LanternObject<torch::TensorList>*)tensors)->get()));
}

void* lantern_embedding_tensor_tensor_intt_bool_bool(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
    return (void *) new LanternObject<torch::Tensor>(torch::embedding(
        ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
}

void* lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
}

void* lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_sparse_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
}

void* lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)offsets)->get(), ((LanternObject<const torch::Tensor &>*)offset2bag)->get(), ((LanternObject<const torch::Tensor &>*)bag_size)->get(), ((LanternObject<const torch::Tensor &>*)maximum_indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<const torch::Tensor &>*)per_sample_weights)->get()));
}

void* lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_sparse_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)offsets)->get(), ((LanternObject<const torch::Tensor &>*)offset2bag)->get(), ((LanternObject<const torch::Tensor &>*)bag_size)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<const torch::Tensor &>*)per_sample_weights)->get()));
}

void* lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(void* size, void* names, void* options, void* memory_format)
{
    return (void *) new LanternObject<torch::Tensor>(torch::empty(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)memory_format)->get()));
}

void* lantern_empty_out_tensor_intarrayref_memoryformat(void* out, void* size, void* memory_format)
{
    return (void *) new LanternObject<torch::Tensor>(torch::empty_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)memory_format)->get()));
}

void* lantern_empty_like_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::empty_like(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_empty_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
    return (void *) new LanternObject<torch::Tensor>(torch::empty_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)memory_format)->get()));
}

void* lantern_eye_intt_tensoroptions(void* n, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_eye_intt_intt_tensoroptions(void* n, void* m, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(void* size, void* fill_value, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::full(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_full_intarrayref_scalar_tensoroptions(void* size, void* fill_value, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::full(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_full_out_tensor_intarrayref_scalar(void* out, void* size, void* fill_value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::full_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get()));
}

void* lantern_full_like_tensor_scalar(void* self, void* fill_value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::full_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get()));
}

void* lantern_full_like_tensor_scalar_tensoroptions(void* self, void* fill_value, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::full_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_grid_sampler_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
    return (void *) new LanternObject<torch::Tensor>(torch::grid_sampler(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
}

void* lantern_hann_window_intt_tensoroptions(void* window_length, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hann_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hamming_window_intt_tensoroptions(void* window_length, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hamming_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hamming_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hamming_window_intt_bool_double_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* beta, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_hinge_embedding_loss_tensor_tensor_double_intt(void* self, void* target, void* margin, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::hinge_embedding_loss(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_group_norm_tensor_intt_tensor_tensor_double_bool(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled)
{
    return (void *) new LanternObject<torch::Tensor>(torch::group_norm(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<int64_t>*)num_groups)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
}

void* lantern__cufft_get_plan_cache_size_intt(void* device_index)
{
    return (void *) new LanternObject<int64_t>(torch::_cufft_get_plan_cache_size(
        ((LanternObject<int64_t>*)device_index)->get()));
}

void* lantern__cufft_get_plan_cache_max_size_intt(void* device_index)
{
    return (void *) new LanternObject<int64_t>(torch::_cufft_get_plan_cache_max_size(
        ((LanternObject<int64_t>*)device_index)->get()));
}

void* lantern__cufft_set_plan_cache_max_size_intt_intt(void* device_index, void* max_size)
{
    torch::_cufft_set_plan_cache_max_size(((LanternObject<int64_t>*)device_index)->get(), ((LanternObject<int64_t>*)max_size)->get());
    return NULL;
}

void* lantern__cufft_clear_plan_cache_intt(void* device_index)
{
    torch::_cufft_clear_plan_cache(((LanternObject<int64_t>*)device_index)->get());
    return NULL;
}

void* lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool(void* self, void* indices, void* values, void* accumulate, void* unsafe)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_index_put_impl_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<torch::TensorList>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)values)->get(), ((LanternObject<bool>*)accumulate)->get(), ((LanternObject<bool>*)unsafe)->get()));
}

void* lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled)
{
    return (void *) new LanternObject<torch::Tensor>(torch::instance_norm(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<const torch::Tensor &>*)running_mean)->get(), ((LanternObject<const torch::Tensor &>*)running_var)->get(), ((LanternObject<bool>*)use_input_stats)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
}

void* lantern_inverse_out_tensor_tensor(void* out, void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::inverse_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_isnan_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::isnan(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_kl_div_tensor_tensor_intt(void* self, void* target, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::kl_div(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable)
{
    return (void *) new LanternObject<torch::Tensor>(torch::layer_norm(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<torch::IntArrayRef>*)normalized_shape)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enable)->get()));
}

void* lantern_linear_tensor_tensor_tensor(void* input, void* weight, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::linear(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_int8_weight_fp32_activation(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)packed)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets)->get(), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_int8_weight(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)packed)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets)->get(), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_fbgemm_pack_gemm_matrix_fp16_tensor(void* input)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_gemm_matrix_fp16(
        ((LanternObject<const torch::Tensor &>*)input)->get()));
}

void* lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_fp16_weight_fp32_activation(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)packed_weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_fp16_weight(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)packed_weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get()));
}

void* lantern_fbgemm_pack_quantized_matrix_tensor(void* input)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<const torch::Tensor &>*)input)->get()));
}

void* lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(void* input, void* K, void* N)
{
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<int64_t>*)K)->get(), ((LanternObject<int64_t>*)N)->get()));
}

void* lantern_linspace_scalar_scalar_intt_tensoroptions(void* start, void* end, void* steps, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::linspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<int64_t>*)steps)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_logspace_scalar_scalar_intt_double_tensoroptions(void* start, void* end, void* steps, void* base, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::logspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<int64_t>*)steps)->get(), ((LanternObject<double>*)base)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_logsumexp_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::margin_ranking_loss(
        ((LanternObject<const torch::Tensor &>*)input1)->get(), ((LanternObject<const torch::Tensor &>*)input2)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_matmul_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
    return (void *) new LanternObject<torch::Tensor>(torch::matmul_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)other)->get()));
}

void* lantern_matrix_rank_tensor_double_bool(void* self, void* tol, void* symmetric)
{
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_rank(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<double>*)tol)->get(), ((LanternObject<bool>*)symmetric)->get()));
}

void* lantern_matrix_rank_tensor_bool(void* self, void* symmetric)
{
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_rank(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)symmetric)->get()));
}

void* lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool1d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
}

void* lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool2d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
}

void* lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool3d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
}

void* lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups)
{
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_convolution(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
}

void* lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_convolution_backward_input(
        ((LanternObject<torch::IntArrayRef>*)self_size)->get(), ((LanternObject<const torch::Tensor &>*)grad_output)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get()));
}

void* lantern__sparse_mm_tensor_tensor(void* sparse, void* dense)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_mm(
        ((LanternObject<const torch::Tensor &>*)sparse)->get(), ((LanternObject<const torch::Tensor &>*)dense)->get()));
}

void* lantern__nnpack_available()
{
    return (void *) new LanternObject<bool>(torch::_nnpack_available(
        ));
}

void* lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref(void* input, void* weight, void* bias, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(void* input, void* grad_output, void* weight, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution_backward_input(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)grad_output)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(void* input, void* weightsize, void* grad_output, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution_backward_weight(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<torch::IntArrayRef>*)weightsize)->get(), ((LanternObject<const torch::Tensor &>*)grad_output)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern_ones_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ones(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_ones_intarrayref_tensoroptions(void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ones(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_ones_out_tensor_intarrayref(void* out, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ones_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_ones_like_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ones_like(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_ones_like_tensor_tensoroptions(void* self, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::ones_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_pairwise_distance_tensor_tensor_double_double_bool(void* x1, void* x2, void* p, void* eps, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::pairwise_distance(
        ((LanternObject<const torch::Tensor &>*)x1)->get(), ((LanternObject<const torch::Tensor &>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_cdist_tensor_tensor_double(void* x1, void* x2, void* p)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cdist(
        ((LanternObject<const torch::Tensor &>*)x1)->get(), ((LanternObject<const torch::Tensor &>*)x2)->get(), ((LanternObject<double>*)p)->get()));
}

void* lantern__cdist_backward_tensor_tensor_tensor_double_tensor(void* grad, void* x1, void* x2, void* p, void* cdist)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_cdist_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)x1)->get(), ((LanternObject<const torch::Tensor &>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<const torch::Tensor &>*)cdist)->get()));
}

void* lantern_pdist_tensor_double(void* self, void* p)
{
    return (void *) new LanternObject<torch::Tensor>(torch::pdist(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get()));
}

void* lantern__pdist_forward_tensor_double(void* self, void* p)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_pdist_forward(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get()));
}

void* lantern__pdist_backward_tensor_tensor_double_tensor(void* grad, void* self, void* p, void* pdist)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_pdist_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<const torch::Tensor &>*)pdist)->get()));
}

void* lantern_cosine_similarity_tensor_tensor_intt_double(void* x1, void* x2, void* dim, void* eps)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cosine_similarity(
        ((LanternObject<const torch::Tensor &>*)x1)->get(), ((LanternObject<const torch::Tensor &>*)x2)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<double>*)eps)->get()));
}

void* lantern_pixel_shuffle_tensor_intt(void* self, void* upscale_factor)
{
    return (void *) new LanternObject<torch::Tensor>(torch::pixel_shuffle(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)upscale_factor)->get()));
}

void* lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(void* input, void* target, void* log_input, void* full, void* eps, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::poisson_nll_loss(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<bool>*)log_input)->get(), ((LanternObject<bool>*)full)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_scalar_tensor_scalar_tensoroptions(void* s, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::scalar_tensor(
        ((LanternObject<torch::Scalar>*)s)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rand_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rand_intarrayref_tensoroptions(void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rand_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rand_out_tensor_intarrayref(void* out, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_rand_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_rand_like_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand_like(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_rand_like_tensor_tensoroptions(void* self, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rand_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_intt_intarrayref_tensoroptions(void* high, void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_intt_intarrayref_generator_tensoroptions(void* high, void* size, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_intt_intt_intarrayref_tensoroptions(void* low, void* high, void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_intt_intt_intarrayref_generator_tensoroptions(void* low, void* high, void* size, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_out_tensor_intt_intarrayref(void* out, void* high, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_randint_out_tensor_intt_intarrayref_generator(void* out, void* high, void* size, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_randint_out_tensor_intt_intt_intarrayref(void* out, void* low, void* high, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_randint_out_tensor_intt_intt_intarrayref_generator(void* out, void* low, void* high, void* size, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_randint_like_tensor_intt(void* self, void* high)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)high)->get()));
}

void* lantern_randint_like_tensor_intt_intt(void* self, void* low, void* high)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get()));
}

void* lantern_randint_like_tensor_intt_tensoroptions(void* self, void* high, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randint_like_tensor_intt_intt_tensoroptions(void* self, void* low, void* high, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randn_intarrayref_tensoroptions(void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randn_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randn_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randn_out_tensor_intarrayref(void* out, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_randn_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_randn_like_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn_like(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_randn_like_tensor_tensoroptions(void* self, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randn_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randperm_intt_tensoroptions(void* n, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randperm_intt_generator_tensoroptions(void* n, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_randperm_out_tensor_intt(void* out, void* n)
{
    return (void *) new LanternObject<torch::Tensor>(torch::randperm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)n)->get()));
}

void* lantern_range_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_range_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_rrelu_tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_rrelu__tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_selu_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::selu(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_selu__tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::selu_(
        ((LanternObject<torch::Tensor &>*)self)->get()));
}

void* lantern_celu_tensor_scalar(void* self, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::celu(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_celu__tensor_scalar(void* self, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::celu_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_stack_tensorlist_intt(void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::stack(
        ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_stack_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::stack_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::sum_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::sum_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_std_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::std_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_std_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::std_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_prod_out_tensor_tensor_intt_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::prod_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_prod_out_tensor_tensor_dimname_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::prod_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)dtype)->get()));
}

void* lantern_tensordot_tensor_tensor_intarrayref_intarrayref(void* self, void* other, void* dims_self, void* dims_other)
{
    return (void *) new LanternObject<torch::Tensor>(torch::tensordot(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)other)->get(), ((LanternObject<torch::IntArrayRef>*)dims_self)->get(), ((LanternObject<torch::IntArrayRef>*)dims_other)->get()));
}

void* lantern_threshold_tensor_scalar_scalar(void* self, void* threshold, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::threshold(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_threshold__tensor_scalar_scalar(void* self, void* threshold, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_(
        ((LanternObject<torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_threshold_out_tensor_tensor_scalar_scalar(void* out, void* self, void* threshold, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_threshold_backward_tensor_tensor_scalar(void* grad_output, void* self, void* threshold)
{
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_backward(
        ((LanternObject<const torch::Tensor &>*)grad_output)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
}

void* lantern_one_hot_tensor_intt(void* self, void* num_classes)
{
    return (void *) new LanternObject<torch::Tensor>(torch::one_hot(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)num_classes)->get()));
}

void* lantern_trapz_tensor_tensor_intt(void* y, void* x, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::trapz(
        ((LanternObject<const torch::Tensor &>*)y)->get(), ((LanternObject<const torch::Tensor &>*)x)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_trapz_tensor_double_intt(void* y, void* dx, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::trapz(
        ((LanternObject<const torch::Tensor &>*)y)->get(), ((LanternObject<double>*)dx)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_trilinear(
        ((LanternObject<const torch::Tensor &>*)i1)->get(), ((LanternObject<const torch::Tensor &>*)i2)->get(), ((LanternObject<const torch::Tensor &>*)i3)->get(), ((LanternObject<torch::IntArrayRef>*)expand1)->get(), ((LanternObject<torch::IntArrayRef>*)expand2)->get(), ((LanternObject<torch::IntArrayRef>*)expand3)->get(), ((LanternObject<torch::IntArrayRef>*)sumdim)->get(), ((LanternObject<int64_t>*)unroll_dim)->get()));
}

void* lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::triplet_margin_loss(
        ((LanternObject<const torch::Tensor &>*)anchor)->get(), ((LanternObject<const torch::Tensor &>*)positive)->get(), ((LanternObject<const torch::Tensor &>*)negative)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)swap)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern__has_compatible_shallow_copy_type_tensor_tensor(void* self, void* from)
{
    return (void *) new LanternObject<bool>(torch::_has_compatible_shallow_copy_type(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)from)->get()));
}

void* lantern__unsafe_view_tensor_intarrayref(void* self, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_unsafe_view(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_var_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::var_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_var_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::var_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_where_tensor(void* condition)
{
    return (void *) new LanternObject<torch::TensorList>(torch::where(
        ((LanternObject<const torch::Tensor &>*)condition)->get()));
}

void* lantern_norm_except_dim_tensor_intt_intt(void* v, void* pow, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::norm_except_dim(
        ((LanternObject<const torch::Tensor &>*)v)->get(), ((LanternObject<int64_t>*)pow)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern__weight_norm_tensor_tensor_intt(void* v, void* g, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_weight_norm(
        ((LanternObject<const torch::Tensor &>*)v)->get(), ((LanternObject<const torch::Tensor &>*)g)->get(), ((LanternObject<int64_t>*)dim)->get()));
}

void* lantern_zeros_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::zeros(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<torch::DimnameList>>*)names)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_zeros_intarrayref_tensoroptions(void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::zeros(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_zeros_out_tensor_intarrayref(void* out, void* size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::zeros_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get()));
}

void* lantern_zeros_like_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::zeros_like(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_zeros_like_tensor_tensoroptions(void* self, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::zeros_like(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern__sparse_sum_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern__sparse_sum_tensor_scalartype(void* self, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
}

void* lantern__sparse_sum_tensor_intarrayref(void* self, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get()));
}

void* lantern__sparse_sum_tensor_intarrayref_scalartype(void* self, void* dim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
}

void* lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)p)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
}

void* lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)p)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)p)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
}

void* lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)p)->get(), ((LanternObject<torch::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_frobenius_norm_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_frobenius_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_nuclear_norm_tensor_bool(void* self, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_nuclear_norm_out_tensor_tensor_bool(void* out, void* self, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_nuclear_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
}

void* lantern_rsub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rsub(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_rsub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rsub(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* sparse, void* dense, void* beta, void* alpha)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_addmm(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)sparse)->get(), ((LanternObject<const torch::Tensor &>*)dense)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
}

void* lantern_sparse_coo_tensor_intarrayref_tensoroptions(void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(void* indices, void* values, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)values)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)values)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_coo_tensor_unsafe(
        ((LanternObject<const torch::Tensor &>*)indices)->get(), ((LanternObject<const torch::Tensor &>*)values)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_to_dense_backward_tensor_tensor(void* grad, void* input)
{
    return (void *) new LanternObject<torch::Tensor>(torch::to_dense_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)input)->get()));
}

void* lantern_to_mkldnn_backward_tensor_tensor(void* grad, void* input)
{
    return (void *) new LanternObject<torch::Tensor>(torch::to_mkldnn_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<const torch::Tensor &>*)input)->get()));
}

void* lantern_meshgrid_tensorlist(void* tensors)
{
    return (void *) new LanternObject<torch::TensorList>(torch::meshgrid(
        ((LanternObject<torch::TensorList>*)tensors)->get()));
}

void* lantern_cartesian_prod_tensorlist(void* tensors)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cartesian_prod(
        ((LanternObject<torch::TensorList>*)tensors)->get()));
}

void* lantern_combinations_tensor_intt_bool(void* self, void* r, void* with_replacement)
{
    return (void *) new LanternObject<torch::Tensor>(torch::combinations(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)r)->get(), ((LanternObject<bool>*)with_replacement)->get()));
}

void* lantern_result_type_tensor_tensor(void* tensor, void* other)
{
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<const torch::Tensor &>*)tensor)->get(), ((LanternObject<const torch::Tensor &>*)other)->get()));
}

void* lantern_result_type_tensor_scalar(void* tensor, void* other)
{
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<const torch::Tensor &>*)tensor)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
}

void* lantern_result_type_scalar_tensor(void* scalar, void* tensor)
{
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Scalar>*)scalar)->get(), ((LanternObject<const torch::Tensor &>*)tensor)->get()));
}

void* lantern_result_type_scalar_scalar(void* scalar1, void* scalar2)
{
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Scalar>*)scalar1)->get(), ((LanternObject<torch::Scalar>*)scalar2)->get()));
}

void* lantern_can_cast_scalartype_scalartype(void* from, void* to)
{
    return (void *) new LanternObject<bool>(torch::can_cast(
        ((LanternObject<torch::ScalarType>*)from)->get(), ((LanternObject<torch::ScalarType>*)to)->get()));
}

void* lantern_promote_types_scalartype_scalartype(void* type1, void* type2)
{
    return (void *) new LanternObject<torch::ScalarType>(torch::promote_types(
        ((LanternObject<torch::ScalarType>*)type1)->get(), ((LanternObject<torch::ScalarType>*)type2)->get()));
}

void* lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::gru_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get()));
}

void* lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rnn_tanh_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get()));
}

void* lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::rnn_relu_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get()));
}

void* lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_gru_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get(), ((LanternObject<const torch::Tensor &>*)packed_ih)->get(), ((LanternObject<const torch::Tensor &>*)packed_hh)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_ih)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
}

void* lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_rnn_relu_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get(), ((LanternObject<const torch::Tensor &>*)packed_ih)->get(), ((LanternObject<const torch::Tensor &>*)packed_hh)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_ih)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
}

void* lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_rnn_tanh_cell(
        ((LanternObject<const torch::Tensor &>*)input)->get(), ((LanternObject<const torch::Tensor &>*)hx)->get(), ((LanternObject<const torch::Tensor &>*)w_ih)->get(), ((LanternObject<const torch::Tensor &>*)w_hh)->get(), ((LanternObject<const torch::Tensor &>*)b_ih)->get(), ((LanternObject<const torch::Tensor &>*)b_hh)->get(), ((LanternObject<const torch::Tensor &>*)packed_ih)->get(), ((LanternObject<const torch::Tensor &>*)packed_hh)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_ih)->get(), ((LanternObject<const torch::Tensor &>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
}

void* lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(void* grad, void* input_size, void* batch_sizes, void* batch_first)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_pack_padded_sequence_backward(
        ((LanternObject<const torch::Tensor &>*)grad)->get(), ((LanternObject<torch::IntArrayRef>*)input_size)->get(), ((LanternObject<const torch::Tensor &>*)batch_sizes)->get(), ((LanternObject<bool>*)batch_first)->get()));
}

void* lantern_cross_out_tensor_tensor_tensor_intt(void* out, void* self, void* other, void* dim)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cross_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
}

void* lantern_index_select_out_tensor_tensor_dimname_tensor(void* out, void* self, void* dim, void* index)
{
    return (void *) new LanternObject<torch::Tensor>(torch::index_select_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<const torch::Tensor &>*)index)->get()));
}

void* lantern_gather_out_tensor_tensor_dimname_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
    return (void *) new LanternObject<torch::Tensor>(torch::gather_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<const torch::Tensor &>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
}

void* lantern__gather_sparse_backward_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* grad)
{
    return (void *) new LanternObject<torch::Tensor>(torch::_gather_sparse_backward(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<const torch::Tensor &>*)index)->get(), ((LanternObject<const torch::Tensor &>*)grad)->get()));
}

void* lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::addcmul_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)tensor1)->get(), ((LanternObject<const torch::Tensor &>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
    return (void *) new LanternObject<torch::Tensor>(torch::addcdiv_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)tensor1)->get(), ((LanternObject<const torch::Tensor &>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
}

void* lantern_cholesky_out_tensor_tensor_bool(void* out, void* self, void* upper)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
}

void* lantern_cholesky_solve_out_tensor_tensor_tensor_bool(void* out, void* self, void* input2, void* upper)
{
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_solve_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
}

void* lantern_lu_solve_out_tensor_tensor_tensor_tensor(void* out, void* self, void* LU_data, void* LU_pivots)
{
    return (void *) new LanternObject<torch::Tensor>(torch::lu_solve_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)LU_data)->get(), ((LanternObject<const torch::Tensor &>*)LU_pivots)->get()));
}

void* lantern_digamma_out_tensor_tensor(void* out, void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::digamma_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_polygamma_out_tensor_intt_tensor(void* out, void* n, void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::polygamma_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_atan2_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
    return (void *) new LanternObject<torch::Tensor>(torch::atan2_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)other)->get()));
}

void* lantern_normal_double_double_intarrayref_generator_tensoroptions(void* mean, void* std, void* size, void* generator, void* options)
{
    return (void *) new LanternObject<torch::Tensor>(torch::normal(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get(), ((LanternObject<const torch::TensorOptions &>*)options)->get()));
}

void* lantern_normal_out_tensor_double_double_intarrayref_generator(void* out, void* mean, void* std, void* size, void* generator)
{
    return (void *) new LanternObject<torch::Tensor>(torch::normal_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<torch::IntArrayRef>*)size)->get(), ((LanternObject<torch::Generator *>*)generator)->get()));
}

void* lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_multilabel_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
}

void* lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
}

void* lantern_nll_loss_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
}

void* lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
}

void* lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)target)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
}

void* lantern_log_sigmoid_out_tensor_tensor(void* out, void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_log_sigmoid_tensor(void* self)
{
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid(
        ((LanternObject<const torch::Tensor &>*)self)->get()));
}

void* lantern_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool2d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<torch::IntArrayRef>*)output_size)->get()));
}

void* lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv2d_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv2d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get()));
}

void* lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get(), ((LanternObject<torch::IntArrayRef>*)dilation)->get()));
}

void* lantern_thnn_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv3d_out(
        ((LanternObject<torch::Tensor &>*)out)->get(), ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

void* lantern_thnn_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv3d(
        ((LanternObject<const torch::Tensor &>*)self)->get(), ((LanternObject<const torch::Tensor &>*)weight)->get(), ((LanternObject<torch::IntArrayRef>*)kernel_size)->get(), ((LanternObject<const torch::Tensor &>*)bias)->get(), ((LanternObject<torch::IntArrayRef>*)stride)->get(), ((LanternObject<torch::IntArrayRef>*)padding)->get()));
}

/* Autogen Body -- End */
