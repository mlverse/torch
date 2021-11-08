#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

int lanternLogEnabled = 0;
void lanternConfigure(int log)
{
  lanternLogEnabled = log;
}

std::string *pLanternLastError = NULL;

const char* lanternVersion()
{
  return "0.1.0.9004";
}

void lanternSetLastError(const char* error)
{
  LLOG("Setting last error to %s", error);
  pLanternLastError = new std::string(error);
}

const char* lanternLastError()
{
  if (pLanternLastError == NULL)
    return NULL;
  else {
    LLOG("Has last error set to %s", pLanternLastError->c_str());
    return pLanternLastError->c_str();
  }
}

void lanternLastErrorClear()
{
  LLOG("Cleared last error");
  pLanternLastError = NULL;
}

void lanternTest()
{
    std::cout << "-- Lantern: 0.1.0" << std::endl;

    std::cout << "-- Testing Tensor" << std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "-- Success!" << std::endl;
}

/* Autogen Body -- Start */
void* _lantern__cast_byte_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Byte(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_char_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Char(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_double_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Double(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_float_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Float(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_int_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Int(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_long_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Long(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_short_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Short(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_half_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cast_Half(
        from_raw::Tensor(self), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool(void* self, void* inputs, void* gradient, void* retain_graph, void* create_graph)
{
  LANTERN_FUNCTION_START
    from_raw::Tensor(self)._backward(((LanternObject<std::vector<torch::Tensor>>*)inputs)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)gradient)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(retain_graph).get())->get(), ((LanternObject<bool>*)create_graph)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set_data_tensor_tensor(void* self, void* new_data)
{
  LANTERN_FUNCTION_START
    from_raw::Tensor(self).set_data(from_raw::Tensor(new_data));
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_data_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).data(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_leaf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_leaf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_output_nr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).output_nr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__version_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self)._version(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_requires_grad__tensor_bool(void* self, void* requires_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).requires_grad_(
        ((LanternObject<bool>*)requires_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_retain_grad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    from_raw::Tensor(self).retain_grad();
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__fw_primal_tensor_intt(void* self, void* level)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self)._fw_primal(
        ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__make_dual_tensor_tensor_intt(void* primal, void* tangent, void* level)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_make_dual(
        from_raw::Tensor(primal), from_raw::Tensor(tangent), ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__unpack_dual_tensor_intt(void* dual, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unpack_dual(
        from_raw::Tensor(dual), ((LanternObject<int64_t>*)level)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename__tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rename_(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rename(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).align_to(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_tensor_dimnamelist_intt(void* self, void* order, void* ellipsis_idx)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).align_to(
        ((LanternPtr<std::vector<torch::Dimname>>*)order)->get(), ((LanternObject<int64_t>*)ellipsis_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).align_as(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_align_tensors_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::align_tensors(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__assert_async_tensor(void* self)
{
  LANTERN_FUNCTION_START
    torch::_assert_async(from_raw::Tensor(self));
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_refine_names_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).refine_names(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_ctc_loss(
        from_raw::Tensor(log_probs), from_raw::Tensor(targets), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* deterministic, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_ctc_loss(
        from_raw::Tensor(log_probs), from_raw::Tensor(targets), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_rnn_flatten_weight()
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_rnn_flatten_weight(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_intt_bool_bool(void* weight_arr, void* weight_stride0, void* input_size, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cudnn_rnn_flatten_weight(
        ((LanternObject<std::vector<torch::Tensor>>*)weight_arr)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<int64_t>*)input_size)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<bool>*)bidirectional)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight_buf)->get(), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)cx)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)dropout_state)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn_backward(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), from_raw::Tensor(weight_buf), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)cx)->get(), from_raw::Tensor(output), ((LanternObject<c10::optional<torch::Tensor>>*)grad_output)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_hy)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_cy)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)dropout_state)->get(), from_raw::Tensor(reserve), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions(void* dropout, void* train, void* dropout_seed, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cudnn_init_dropout_state(
        ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<int64_t>*)dropout_seed)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__debug_has_internal_overlap_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::_debug_has_internal_overlap(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__fused_dropout_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fused_dropout(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__masked_scale_tensor_tensor_double(void* self, void* mask, void* scale)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_masked_scale(
        from_raw::Tensor(self), from_raw::Tensor(mask), ((LanternObject<double>*)scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype(void* quasi, void* n, void* sobolstate, void* dimension, void* num_generated, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_sobol_engine_draw(
        from_raw::Tensor(quasi), ((LanternObject<int64_t>*)n)->get(), from_raw::Tensor(sobolstate), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(void* self, void* n, void* sobolstate, void* dimension, void* num_generated)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sobol_engine_ff_(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get(), from_raw::Tensor(sobolstate), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_scramble__tensor_tensor_intt(void* self, void* ltm, void* dimension)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sobol_engine_scramble_(
        from_raw::Tensor(self), from_raw::Tensor(ltm), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_initialize_state__tensor_intt(void* self, void* dimension)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sobol_engine_initialize_state_(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__reshape_from_tensor_tensor_tensor(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_reshape_from_tensor(
        from_raw::Tensor(self), from_raw::Tensor(shape)));
  LANTERN_FUNCTION_END
}

void* _lantern__shape_as_tensor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_shape_as_tensor(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dropout(
        from_raw::Tensor(input), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dropout_(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::feature_dropout(
        from_raw::Tensor(input), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::feature_dropout_(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::alpha_dropout(
        from_raw::Tensor(input), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::alpha_dropout_(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::feature_alpha_dropout(
        from_raw::Tensor(input), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::feature_alpha_dropout_(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::abs(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).abs(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::abs_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).abs_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::abs_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::absolute(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).absolute(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).absolute_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::absolute_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::angle(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_angle_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).angle(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::angle_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_real_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::view_as_real(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::view_as_complex(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sgn(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sgn(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sgn_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sgn_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_real_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::real(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_imag_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::imag(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conj(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).conj(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conj_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_conj(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acos(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).acos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acos_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).acos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acos_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccos(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arccos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccos_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arccos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccos_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool1d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool1d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_add_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::add(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).add(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).add_(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::add_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_add_relu(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_add_relu_(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_add_relu_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::add(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).add(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).add_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addmv(
        from_raw::Tensor(self), from_raw::Tensor(mat), from_raw::Tensor(vec), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv_tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addmv(
        from_raw::Tensor(mat), from_raw::Tensor(vec), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv__tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addmv_(
        from_raw::Tensor(self), from_raw::Tensor(mat), from_raw::Tensor(vec), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv__tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addmv_(
        from_raw::Tensor(mat), from_raw::Tensor(vec), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addmv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat), from_raw::Tensor(vec), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addr(
        from_raw::Tensor(self), from_raw::Tensor(vec1), from_raw::Tensor(vec2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addr(
        from_raw::Tensor(vec1), from_raw::Tensor(vec2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr__tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addr_(
        from_raw::Tensor(vec1), from_raw::Tensor(vec2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addr_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(vec1), from_raw::Tensor(vec2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_tensor_intarrayref_bool(void* theta, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::affine_grid_generator(
        from_raw::Tensor(theta), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_backward_tensor_intarrayref_bool(void* grad, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::affine_grid_generator_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::all(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).all(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::all_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::all(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).all(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::all_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_allclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::allclose(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_allclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).allclose(
        from_raw::Tensor(other), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::any(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).any(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::any_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::any(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).any(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::any_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_tensoroptions(void* end, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arange(
        ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_tensor_scalar(void* out, void* end)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arange_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)end)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arange_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dim_arange_tensor_intt(void* like, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_dim_arange(
        from_raw::Tensor(like), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argmax(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).argmax(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmax_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argmax_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmin_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argmin(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmin_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).argmin(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmin_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argmin_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acosh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).acosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acosh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).acosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::acosh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccosh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arccosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccosh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arccosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arccosh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asinh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).asinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asinh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).asinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asinh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsinh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arcsinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsinh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arcsinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsinh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atanh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atanh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atanh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctanh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arctanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctanh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arctanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctanh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided_tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::as_strided(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided_tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).as_strided(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided__tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::as_strided_(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided__tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).as_strided_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asin(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).asin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asin_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).asin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::asin_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsin(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arcsin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsin_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arcsin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arcsin_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atan(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atan_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atan_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctan(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arctan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctan_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).arctan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::arctan_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_1d_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atleast_1d(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_1d_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::atleast_1d(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_2d_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atleast_2d(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_2d_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::atleast_2d(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_3d_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atleast_3d(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_3d_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::atleast_3d(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_baddbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::baddbmm(
        from_raw::Tensor(self), from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).baddbmm(
        from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).baddbmm_(
        from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_baddbmm_mkl_(
        from_raw::Tensor(self), from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::baddbmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::batch_norm(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_batch_norm_tensor_tensor_tensor_tensor_tensor_double_double_intt(void* input, void* weight, void* bias, void* mean, void* var, void* eps, void* output_scale, void* output_zero_point)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_batch_norm(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), from_raw::Tensor(mean), from_raw::Tensor(var), ((LanternObject<double>*)eps)->get(), ((LanternObject<double>*)output_scale)->get(), ((LanternObject<int64_t>*)output_zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool_tensor(void* impl_index, void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var_transform, void* train, void* eps, void* output_mask, void* reservedSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index_backward(
        ((LanternObject<int64_t>*)impl_index)->get(), from_raw::Tensor(input), from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_var_transform)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get(), from_raw::Tensor(reservedSpace))));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bernoulli(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bernoulli(
        ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_out_tensor_tensor_generator(void* out, void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bernoulli_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__tensor_tensor_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bernoulli_(
        from_raw::Tensor(p), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bernoulli_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bernoulli(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bernoulli(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bilinear_tensor_tensor_tensor_tensor(void* input1, void* input2, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bilinear(
        from_raw::Tensor(input1), from_raw::Tensor(input2), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy_with_logits(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)pos_weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binary_cross_entropy_with_logits_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)pos_weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bincount_tensor_tensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bincount(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)weights)->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bincount_tensor_tensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bincount(
        ((LanternObject<c10::optional<torch::Tensor>>*)weights)->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_not(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_not_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::copysign_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::copysign(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).copysign(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).copysign_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::copysign(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).copysign(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).copysign_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::copysign_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_not(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_not_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_xor(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_xor(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_xor_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_xor_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_and(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_and(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_and_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_and_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_or(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_or(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logical_or_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logical_or_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bmm(
        from_raw::Tensor(self), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bmm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bmm(
        from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_tensor_tensor_bool(void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_bmm(
        from_raw::Tensor(self), from_raw::Tensor(mat2), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_out_tensor_tensor_tensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_out_tensor_tensor_tensor_bool(void* out, void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_bmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat2), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_broadcast_tensors_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::broadcast_tensors(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_broadcast_to_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::broadcast_to(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_broadcast_to_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).broadcast_to(
        ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cat_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_tensorlist_dimname(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_tensor_tensorlist_dimname(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cat_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_block_diag_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::block_diag(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ceil(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ceil(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ceil_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ceil_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ceil_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_chain_matmul_tensorlist(void* matrices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::chain_matmul(
        ((LanternObject<std::vector<torch::Tensor>>*)matrices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chain_matmul_out_tensor_tensorlist(void* out, void* matrices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::chain_matmul_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)matrices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_chunk(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).unsafe_chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::chunk(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_tensor_intt_intt(void* self, void* sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::tensor_split(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_tensor_intt_intt(void* self, void* sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).tensor_split(
        ((LanternObject<int64_t>*)sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_tensor_intarrayref_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::tensor_split(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_tensor_intarrayref_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).tensor_split(
        ((LanternObject<std::vector<int64_t>>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_tensor_tensor_intt(void* self, void* tensor_indices_or_sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::tensor_split(
        from_raw::Tensor(self), from_raw::Tensor(tensor_indices_or_sections), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_tensor_tensor_intt(void* self, void* tensor_indices_or_sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).tensor_split(
        from_raw::Tensor(tensor_indices_or_sections), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp(
        ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp__tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp__tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_(
        ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_out_tensor_tensor_tensor_tensor(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max_tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_max(
        ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_tensor_tensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max(
        from_raw::Tensor(self), from_raw::Tensor(max)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max_tensor_tensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_max(
        from_raw::Tensor(max)));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max__tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max__tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_max_(
        ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max__tensor_tensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max_(
        from_raw::Tensor(self), from_raw::Tensor(max)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max__tensor_tensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_max_(
        from_raw::Tensor(max)));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_out_tensor_tensor_scalar(void* out, void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_out_tensor_tensor_tensor(void* out, void* self, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_max_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(max)));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min_tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_min(
        ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_tensor_tensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min(
        from_raw::Tensor(self), from_raw::Tensor(min)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min_tensor_tensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_min(
        from_raw::Tensor(min)));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min__tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min__tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_min_(
        ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min__tensor_tensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min_(
        from_raw::Tensor(self), from_raw::Tensor(min)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min__tensor_tensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clamp_min_(
        from_raw::Tensor(min)));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_out_tensor_tensor_scalar(void* out, void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_out_tensor_tensor_tensor(void* out, void* self, void* min)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clamp_min_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(min)));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clip(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip_tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clip(
        ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clip_(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip__tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip__tensor_tensor_tensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clip_(
        ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_out_tensor_tensor_tensor_tensor(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clip_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Tensor>>*)min)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_is_acceptable_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::cudnn_is_acceptable(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_tensor_tensor(void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::complex(
        from_raw::Tensor(real), from_raw::Tensor(imag)));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_out_tensor_tensor_tensor(void* out, void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::complex_out(
        from_raw::Tensor(out), from_raw::Tensor(real), from_raw::Tensor(imag)));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_tensor_tensor(void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::polar(
        from_raw::Tensor(abs), from_raw::Tensor(angle)));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_out_tensor_tensor_tensor(void* out, void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::polar_out(
        from_raw::Tensor(out), from_raw::Tensor(abs), from_raw::Tensor(angle)));
  LANTERN_FUNCTION_END
}

void* _lantern_constant_pad_nd_tensor_intarrayref_scalar(void* self, void* pad, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::constant_pad_nd(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)pad)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_contiguous_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).contiguous(
        ((LanternObject<torch::MemoryFormat>*)memory_format)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::convolution(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::convolution_overrideable(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool(void* grad_output, void* input, void* weight, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::convolution_backward_overrideable(
        from_raw::Tensor(grad_output), from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_convolution(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_convolution(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_mode_tensor_tensor_tensor_intarrayref_stdstring_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_convolution_mode(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_convolution_nogroup(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool_stdarraybool(void* ggI, void* ggW, void* ggb, void* gO, void* weight, void* self, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_convolution_double_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)ggI)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)ggW)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)ggb)->get(), from_raw::Tensor(gO), from_raw::Tensor(weight), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv1d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv2d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv3d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv1d_tensor_tensor_tensor_intarrayref_stdstring_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv1d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv2d_tensor_tensor_tensor_intarrayref_stdstring_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv2d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv3d_tensor_tensor_tensor_intarrayref_stdstring_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv3d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_tensor_tensor_tensor_intt(void* self, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv_tbc(
        from_raw::Tensor(self), from_raw::Tensor(weight), from_raw::Tensor(bias), ((LanternObject<int64_t>*)pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt(void* self, void* input, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_tbc_backward(
        from_raw::Tensor(self), from_raw::Tensor(input), from_raw::Tensor(weight), from_raw::Tensor(bias), ((LanternObject<int64_t>*)pad)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv_transpose1d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv_transpose2d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv_transpose3d(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copy__tensor_tensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).copy_(
        from_raw::Tensor(src), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__copy_from_tensor_tensor_bool(void* self, void* dst, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_copy_from(
        from_raw::Tensor(self), from_raw::Tensor(dst), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cos(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cos_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cos_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cosh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cosh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cosh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cosine_embedding_loss(
        from_raw::Tensor(input1), from_raw::Tensor(input2), from_raw::Tensor(target), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::count_nonzero(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).count_nonzero(
        ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::count_nonzero(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).count_nonzero(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt(void* theta, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_affine_grid_generator(
        from_raw::Tensor(theta), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt(void* grad, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_affine_grid_generator_backward(
        from_raw::Tensor(grad), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_tensor(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon, void* reserveSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm_backward(
        from_raw::Tensor(input), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_var)->get(), ((LanternObject<double>*)epsilon)->get(), from_raw::Tensor(reserveSpace))));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_transpose(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_transpose(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_transpose(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_transpose_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_transpose_backward_input(
        from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_transpose_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_relu_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_relu(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_add_relu_tensor_tensor_tensor_scalar_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* z, void* alpha, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_convolution_add_relu(
        from_raw::Tensor(self), from_raw::Tensor(weight), from_raw::Tensor(z), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(alpha).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_tensor_tensor(void* self, void* grid)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cudnn_grid_sampler(
        from_raw::Tensor(self), from_raw::Tensor(grid)));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor(void* self, void* grid, void* grad_output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_grid_sampler_backward(
        from_raw::Tensor(self), from_raw::Tensor(grid), from_raw::Tensor(grad_output))));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).cummax(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_tensor_tensor_tensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).cummax(
        ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_tensor_tensor_tensor_dimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummax_helper_tensor_tensor_tensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummax_helper(from_raw::Tensor(self), from_raw::Tensor(values), from_raw::Tensor(indices), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).cummin(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_tensor_tensor_tensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).cummin(
        ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_tensor_tensor_tensor_dimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummin_helper_tensor_tensor_tensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummin_helper(from_raw::Tensor(self), from_raw::Tensor(values), from_raw::Tensor(indices), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummaxmin_backward_tensor_tensor_tensor_intt(void* grad, void* input, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cummaxmin_backward(
        from_raw::Tensor(grad), from_raw::Tensor(input), from_raw::Tensor(indices), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumprod(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumprod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod__tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumprod_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumprod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumprod(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumprod(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod__tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumprod_(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumprod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_backward_tensor_tensor_intt_tensor(void* grad, void* input, void* dim, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumprod_backward(
        from_raw::Tensor(grad), from_raw::Tensor(input), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumsum(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumsum(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum__tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumsum_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumsum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumsum(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumsum(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum__tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cumsum_(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cumsum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ctc_loss(
        from_raw::Tensor(log_probs), from_raw::Tensor(targets), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ctc_loss(
        from_raw::Tensor(log_probs), from_raw::Tensor(targets), from_raw::Tensor(input_lengths), from_raw::Tensor(target_lengths), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_ctc_loss(
        from_raw::Tensor(log_probs), from_raw::Tensor(targets), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool(void* grad, void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* neg_log_likelihood, void* log_alpha, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_ctc_loss_backward(
        from_raw::Tensor(grad), from_raw::Tensor(log_probs), from_raw::Tensor(targets), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), from_raw::Tensor(neg_log_likelihood), from_raw::Tensor(log_alpha), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_embed_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diag_embed(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_embed_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diag_embed(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagflat_tensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diagflat(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagflat_tensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diagflat(
        ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diagonal(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diagonal(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_tensor_dimname_dimname_dimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diagonal(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)outdim)->get(), ((LanternObject<torch::Dimname>*)dim1)->get(), ((LanternObject<torch::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_tensor_dimname_dimname_dimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diagonal(
        ((LanternObject<torch::Dimname>*)outdim)->get(), ((LanternObject<torch::Dimname>*)dim1)->get(), ((LanternObject<torch::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_backward_tensor_intarrayref_intt_intt_intt(void* grad, void* input_sizes, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diagonal_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill_diagonal__tensor_scalar_bool(void* self, void* fill_value, void* wrap)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fill_diagonal_(
        ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<bool>*)wrap)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diff_tensor_intt_intt_tensor_tensor(void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diff(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)prepend)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)append)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diff_tensor_intt_intt_tensor_tensor(void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diff(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)prepend)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)append)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diff_out_tensor_tensor_intt_intt_tensor_tensor(void* out, void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diff_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)prepend)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)append)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_scalar_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(spacing).get())->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_scalar_intarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)spacing)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_intarrayref_intt(void* self, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_arrayrefscalar_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)spacing)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_arrayrefscalar_intarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)spacing)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_tensorlist_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<std::vector<torch::Tensor>>*)spacing)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_tensor_tensorlist_intarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::gradient(
        from_raw::Tensor(self), ((LanternObject<std::vector<torch::Tensor>>*)spacing)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_div_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div(
        from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div_(
        from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_out_tensor_tensor_tensor_stdstring(void* out, void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::div(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).div_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide(
        from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_tensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide_(
        from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_out_tensor_tensor_tensor_stdstring(void* out, void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::divide(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_scalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).divide_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::true_divide(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return make_unique::Tensor(from_raw::Tensor(self).true_divide(
        from_raw::Tensor(other)));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return make_unique::Tensor(from_raw::Tensor(self).true_divide_(
        from_raw::Tensor(other)));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::true_divide_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::true_divide(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return make_unique::Tensor(from_raw::Tensor(self).true_divide(
        ((LanternObject<torch::Scalar>*)other)->get()));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return make_unique::Tensor(from_raw::Tensor(self).true_divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_dot_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dot(
        from_raw::Tensor(self), from_raw::Tensor(tensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dot_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).dot(
        from_raw::Tensor(tensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_dot_out_tensor_tensor_tensor(void* out, void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dot_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(tensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::vdot(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vdot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).vdot(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::vdot_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_einsum_stdstring_tensorlist(void* equation, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::einsum(
        ((LanternObject<std::string>*)equation)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_tensor_tensor_intt_bool_bool(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::embedding(
        from_raw::Tensor(weight), from_raw::Tensor(indices), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::embedding_backward(
        from_raw::Tensor(grad), from_raw::Tensor(indices), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool(void* grad_output, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::embedding_dense_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(indices), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_renorm__tensor_tensor_double_double(void* self, void* indices, void* max_norm, void* norm_type)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::embedding_renorm_(
        from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<double>*)max_norm)->get(), ((LanternObject<double>*)norm_type)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::embedding_sparse_backward(
        from_raw::Tensor(grad), from_raw::Tensor(indices), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_forward_only_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag_forward_only(
        from_raw::Tensor(weight), from_raw::Tensor(indices), from_raw::Tensor(offsets), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<int64_t>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__rowwise_prune_tensor_tensor_scalartype(void* weight, void* mask, void* compressed_indices_dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_rowwise_prune(
        from_raw::Tensor(weight), from_raw::Tensor(mask), ((LanternObject<torch::ScalarType>*)compressed_indices_dtype)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_row_stack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::row_stack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_row_stack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::row_stack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::embedding_bag(
        from_raw::Tensor(weight), from_raw::Tensor(indices), from_raw::Tensor(offsets), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<bool>*)include_last_offset)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::embedding_bag(
        from_raw::Tensor(weight), from_raw::Tensor(indices), from_raw::Tensor(offsets), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<c10::optional<int64_t>>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag(
        from_raw::Tensor(weight), from_raw::Tensor(indices), from_raw::Tensor(offsets), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<int64_t>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor_intt(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_embedding_bag_backward(
        from_raw::Tensor(grad), from_raw::Tensor(indices), from_raw::Tensor(offsets), from_raw::Tensor(offset2bag), from_raw::Tensor(bag_size), from_raw::Tensor(maximum_indices), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor_intt(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_embedding_bag_sparse_backward(
        from_raw::Tensor(grad), from_raw::Tensor(indices), from_raw::Tensor(offsets), from_raw::Tensor(offset2bag), from_raw::Tensor(bag_size), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor_intt(void* grad, void* indices, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_embedding_bag_dense_backward(
        from_raw::Tensor(grad), from_raw::Tensor(indices), from_raw::Tensor(offset2bag), from_raw::Tensor(bag_size), from_raw::Tensor(maximum_indices), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)per_sample_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt_intt(void* grad, void* weight, void* indices, void* offsets, void* offset2bag, void* mode, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_embedding_bag_per_sample_weights_backward(
        from_raw::Tensor(grad), from_raw::Tensor(weight), from_raw::Tensor(indices), from_raw::Tensor(offsets), from_raw::Tensor(offset2bag), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(void* size, void* names, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_intarrayref_tensoroptions_memoryformat(void* size, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_empty_tensor_intarrayref_tensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).new_empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_empty_strided_tensor_intarrayref_intarrayref_tensoroptions(void* self, void* size, void* stride, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).new_empty_strided(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_full_tensor_intarrayref_scalar_tensoroptions(void* self, void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).new_full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_zeros_tensor_intarrayref_tensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).new_zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat(void* size, void* options, void* scale, void* zero_point, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_empty_affine_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat(void* size, void* scales, void* zero_points, void* axis, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_empty_per_channel_affine_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), from_raw::Tensor(scales), from_raw::Tensor(zero_points), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize__tensor_intarrayref_memoryformat(void* self, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).resize_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_quantized_intarrayref_tensor(void* size, void* qtensor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), from_raw::Tensor(qtensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_out_tensor_intarrayref_memoryformat(void* out, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty_like(
        from_raw::Tensor(self), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_strided_intarrayref_intarrayref_tensoroptions(void* size, void* stride, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::empty_strided(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erf(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erf_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erf_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erf_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erfc(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erfc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erfc_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erfc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erfc_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).exp_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp2(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).exp2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp2_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).exp2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::exp2_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::expm1(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).expm1(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::expm1_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).expm1_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::expm1_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_tensor_intarrayref_bool(void* self, void* size, void* implicit)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).expand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)implicit)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).expand_as(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_tensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_intt_tensoroptions(void* n, void* m, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_tensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eye_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_tensor_intt_intt(void* out, void* n, void* m)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eye_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flatten(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_intt_intt_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flatten(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_intt_intt_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_dimname_dimname_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flatten(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)start_dim)->get(), ((LanternObject<torch::Dimname>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_dimname_dimname_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flatten(
        ((LanternObject<torch::Dimname>*)start_dim)->get(), ((LanternObject<torch::Dimname>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_dimnamelist_dimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flatten(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dims)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_dimnamelist_dimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flatten(
        ((LanternPtr<std::vector<torch::Dimname>>*)dims)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_tensor_intt_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).unflatten(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_tensor_dimname_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).unflatten(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__tensor_scalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fill_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__tensor_scalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fill_(
        ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__tensor_tensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fill_(
        from_raw::Tensor(self), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__tensor_tensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fill_(
        from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor_divide(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor_divide(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor_divide_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor_divide_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::floor_divide(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor_divide(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).floor_divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frac(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).frac(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frac_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).frac_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frac_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(void* size, void* fill_value, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_intarrayref_scalar_tensoroptions(void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_out_tensor_intarrayref_scalar(void* out, void* size, void* fill_value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::full_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_like_tensor_scalar_tensoroptions_memoryformat(void* self, void* fill_value, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::full_like(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_from_file_stdstring_bool_intt_tensoroptions(void* filename, void* shared, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::from_file(
        ((LanternObject<std::string>*)filename)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(shared).get())->get(), ((LanternObject<c10::optional<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gcd_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gcd(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gcd(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gcd_(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gcd_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lcm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lcm(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lcm(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lcm_(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lcm_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::grid_sampler(
        from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::grid_sampler_2d(
        from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_grid_sampler_2d_cpu_fallback(
        from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_grid_sampler_2d_cpu_fallback_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::grid_sampler_3d(
        from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(input), from_raw::Tensor(grid), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hinge_embedding_loss_tensor_tensor_double_intt(void* self, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hinge_embedding_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_group_norm_tensor_intt_tensor_tensor_double_bool(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::group_norm(
        from_raw::Tensor(input), ((LanternObject<int64_t>*)num_groups)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_tensor_tensor_tensor_intt_intt_intt_intt_double(void* input, void* weight, void* bias, void* N, void* C, void* HxW, void* group, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_stdarraybool(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* N, void* C, void* HxW, void* group, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm_backward(
        from_raw::Tensor(grad_out), from_raw::Tensor(input), from_raw::Tensor(mean), from_raw::Tensor(rstd), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_r2c_tensor_intarrayref_intt_bool(void* self, void* dim, void* normalization, void* onesided)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_r2c(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_r2c_out_tensor_tensor_intarrayref_intt_bool(void* out, void* self, void* dim, void* normalization, void* onesided)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_r2c_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2r_tensor_intarrayref_intt_intt(void* self, void* dim, void* normalization, void* last_dim_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_c2r(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<int64_t>*)last_dim_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2r_out_tensor_tensor_intarrayref_intt_intt(void* out, void* self, void* dim, void* normalization, void* last_dim_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_c2r_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<int64_t>*)last_dim_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2c_tensor_intarrayref_intt_bool(void* self, void* dim, void* normalization, void* forward)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_c2c(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)forward)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2c_out_tensor_tensor_intarrayref_intt_bool(void* out, void* self, void* dim, void* normalization, void* forward)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fft_c2c_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)forward)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cufft_get_plan_cache_size_intt(void* device_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::_cufft_get_plan_cache_size(
        ((LanternObject<int64_t>*)device_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cufft_get_plan_cache_max_size_intt(void* device_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::_cufft_get_plan_cache_max_size(
        ((LanternObject<int64_t>*)device_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cufft_set_plan_cache_max_size_intt_intt(void* device_index, void* max_size)
{
  LANTERN_FUNCTION_START
    torch::_cufft_set_plan_cache_max_size(((LanternObject<int64_t>*)device_index)->get(), ((LanternObject<int64_t>*)max_size)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__cufft_clear_plan_cache_intt(void* device_index)
{
  LANTERN_FUNCTION_START
    torch::_cufft_clear_plan_cache(((LanternObject<int64_t>*)device_index)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_index_tensor_constclistcoptionaltensor(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index(
        from_raw::Tensor(self), ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_tensor_constclistcoptionaltensor(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index(
        ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_copy_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_copy(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_copy(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_copy_(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_copy(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_copy(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put__tensor_constclistcoptionaltensor_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_put_(
        from_raw::Tensor(self), ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get(), from_raw::Tensor(values), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put__tensor_constclistcoptionaltensor_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_put_(
        ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get(), from_raw::Tensor(values), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put_tensor_constclistcoptionaltensor_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_put(
        from_raw::Tensor(self), ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get(), from_raw::Tensor(values), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put_tensor_constclistcoptionaltensor_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_put(
        ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get(), from_raw::Tensor(values), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__index_put_impl__tensor_constclistcoptionaltensor_tensor_bool_bool(void* self, void* indices, void* values, void* accumulate, void* unsafe)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_index_put_impl_(
        from_raw::Tensor(self), ((LanternObject<c10::List<c10::optional<torch::Tensor>>>*)indices)->get(), from_raw::Tensor(values), ((LanternObject<bool>*)accumulate)->get(), ((LanternObject<bool>*)unsafe)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::instance_norm(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)use_input_stats)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::inverse(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_inverse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).inverse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::inverse_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__inverse_helper_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_inverse_helper(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_isclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isclose(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isclose(
        from_raw::Tensor(other), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isnan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isnan(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isnan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isnan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_distributed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_distributed(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_distributed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_distributed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_floating_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_floating_point(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_floating_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_floating_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_complex(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_complex(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isreal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isreal(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isreal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isreal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_nonzero(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_same_size_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_same_size(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_same_size_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_same_size(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_is_signed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_signed(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_signed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_signed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_tensor_tensor_intt_bool(void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kl_div(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_backward_tensor_tensor_tensor_intt_bool(void* grad_output, void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kl_div_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kron_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kron(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kron_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).kron(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_kron_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::kron_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_tensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_tensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_tensor_intt_dimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_tensor_intt_dimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::layer_norm(
        from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)normalized_shape)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enable)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_tensor_intarrayref_tensor_tensor_double(void* input, void* normalized_shape, void* weight, void* bias, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm(
        from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)normalized_shape)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_backward_tensor_tensor_intarrayref_tensor_tensor_tensor_tensor_stdarraybool(void* grad_out, void* input, void* normalized_shape, void* mean, void* rstd, void* weight, void* bias, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm_backward(
        from_raw::Tensor(grad_out), from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)normalized_shape)->get(), from_raw::Tensor(mean), from_raw::Tensor(rstd), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num_tensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nan_to_num(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nan_to_num_tensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nan_to_num(
        ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num__tensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nan_to_num_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nan_to_num__tensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nan_to_num_(
        ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num_out_tensor_tensor_double_double_double(void* out, void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nan_to_num_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linear_tensor_tensor_tensor(void* input, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linear(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_tensor_tensor_tensor(void* self, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_linear(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_input_intarrayref_tensor_tensor(void* input_size, void* grad_output, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_linear_backward_input(
        ((LanternObject<std::vector<int64_t>>*)input_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_weights_tensor_tensor_tensor_bool(void* grad_output, void* input, void* weight, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_linear_backward_weights(
        from_raw::Tensor(grad_output), from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<bool>*)bias_defined)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_tensor_tensor_tensor_stdarraybool(void* self, void* grad_output, void* weight, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_linear_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_linear_int8_weight_fp32_activation(
        from_raw::Tensor(input), from_raw::Tensor(weight), from_raw::Tensor(packed), from_raw::Tensor(col_offsets), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), from_raw::Tensor(bias)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_linear_int8_weight(
        from_raw::Tensor(input), from_raw::Tensor(weight), from_raw::Tensor(packed), from_raw::Tensor(col_offsets), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), from_raw::Tensor(bias)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_quantize_weight_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fbgemm_linear_quantize_weight(
        from_raw::Tensor(input))));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_gemm_matrix_fp16_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_pack_gemm_matrix_fp16(
        from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_linear_fp16_weight_fp32_activation(
        from_raw::Tensor(input), from_raw::Tensor(packed_weight), from_raw::Tensor(bias)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_linear_fp16_weight(
        from_raw::Tensor(input), from_raw::Tensor(packed_weight), from_raw::Tensor(bias)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_pack_quantized_matrix(
        from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(void* input, void* K, void* N)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fbgemm_pack_quantized_matrix(
        from_raw::Tensor(input), ((LanternObject<int64_t>*)K)->get(), ((LanternObject<int64_t>*)N)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ldexp(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ldexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ldexp(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ldexp_(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ldexp__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ldexp_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ldexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_scalar_scalar_intt_tensoroptions(void* start, void* end, void* steps, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_out_tensor_scalar_scalar_intt(void* out, void* start, void* end, void* steps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linspace_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log10(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log10(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log10_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log10_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log10_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log1p(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log1p(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log1p_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log1p_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log1p_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log2(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log2_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log2_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logaddexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logaddexp(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logaddexp(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logaddexp2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logaddexp2(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logaddexp2(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).xlogy(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_scalar_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy(
        ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).xlogy(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy_(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).xlogy_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).xlogy_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_tensor_scalar_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::xlogy_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logdet(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logdet(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_scalar_scalar_intt_double_tensoroptions(void* start, void* end, void* steps, void* base, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_out_tensor_scalar_scalar_intt_double(void* out, void* start, void* end, void* steps, void* base)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logspace_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log_softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_softmax(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log_softmax(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_log_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_log_softmax_backward_data(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_logcumsumexp(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_logcumsumexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logcumsumexp(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logcumsumexp(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logcumsumexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logcumsumexp(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logcumsumexp(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_tensor_tensor_dimname(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logcumsumexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logsumexp(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logsumexp(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logsumexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logsumexp(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logsumexp(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logsumexp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::margin_ranking_loss(
        from_raw::Tensor(input1), from_raw::Tensor(input2), from_raw::Tensor(target), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matmul(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matmul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).matmul(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matmul_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_tensor_double_bool(void* self, void* tol, void* symmetric)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_rank(
        from_raw::Tensor(self), ((LanternObject<double>*)tol)->get(), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_tensor_bool(void* self, void* symmetric)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_rank(
        from_raw::Tensor(self), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_power_tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_power(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_power_tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).matrix_power(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_power_out_tensor_tensor_intt(void* out, void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_power_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_exp(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).matrix_exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_backward_tensor_tensor(void* self, void* grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::matrix_exp_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad)));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_tensor_tensor(void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_compute_linear_combination(
        from_raw::Tensor(input), from_raw::Tensor(coefficients)));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_out_tensor_tensor_tensor(void* out, void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_compute_linear_combination_out(
        from_raw::Tensor(out), from_raw::Tensor(input), from_raw::Tensor(coefficients)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).max(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor_intt_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        from_raw::Tensor(max), from_raw::Tensor(max_values), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).max(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor_dimname_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        from_raw::Tensor(max), from_raw::Tensor(max_values), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_value_selecting_reduction_backward_tensor_intt_tensor_intarrayref_bool(void* grad, void* dim, void* indices, void* sizes, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::value_selecting_reduction_backward(
        from_raw::Tensor(grad), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::amax(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amax_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).amax(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::amax_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool1d_with_indices(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_max_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* grad_output, void* output, void* input, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_max_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_max_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* grad_output, void* output, void* input, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_max_pool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_max_pool1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_max_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mean(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mean(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mean(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mean(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mean_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mean(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mean(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mean_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::median(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).median(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).median(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).median(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanmedian(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nanmedian(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).nanmedian(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).nanmedian(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).min(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor_intt_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        from_raw::Tensor(min), from_raw::Tensor(min_indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).min(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor_dimname_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        from_raw::Tensor(min), from_raw::Tensor(min_indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::amin(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amin_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).amin(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::amin_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward_weights(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm_backward(
        from_raw::Tensor(input), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_var)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_bias_tensor(void* grad_output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_backward_bias(
        from_raw::Tensor(grad_output)));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_transpose(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_transpose_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_transpose_backward_input(
        from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_convolution_transpose_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_depthwise_convolution(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_depthwise_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_depthwise_convolution_backward(
        from_raw::Tensor(self), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::miopen_depthwise_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)cx)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)dropout_state)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn_backward(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), from_raw::Tensor(weight_buf), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)cx)->get(), from_raw::Tensor(output), ((LanternObject<c10::optional<torch::Tensor>>*)grad_output)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_hy)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_cy)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)dropout_state)->get(), from_raw::Tensor(reserve), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mm(
        from_raw::Tensor(self), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mm(
        from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_out_tensor_tensor_tensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_mm_tensor_tensor(void* sparse, void* dense)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_mm(
        from_raw::Tensor(sparse), from_raw::Tensor(dense)));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sparse_matmul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sparse_matmul(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_mask_helper_tensor_tensor(void* t, void* mask_indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_mask_helper(
        from_raw::Tensor(t), from_raw::Tensor(mask_indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).mode(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).mode(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mul(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mul(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mul_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mul_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mul(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mul(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mul_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multiply(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).multiply(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).multiply_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multiply_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multiply(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).multiply(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).multiply_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_tensor_tensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mv(
        from_raw::Tensor(self), from_raw::Tensor(vec)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mv_tensor_tensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mv(
        from_raw::Tensor(vec)));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_out_tensor_tensor_tensor(void* out, void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(vec)));
  LANTERN_FUNCTION_END
}

void* _lantern_mvlgamma_tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mvlgamma(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma_tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mvlgamma(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma__tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).mvlgamma_(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_copy_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::narrow_copy(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_copy_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).narrow_copy(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_copy_out_tensor_tensor_intt_intt_intt(void* out, void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::narrow_copy_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::narrow(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).narrow(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_tensor_intt_tensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::narrow(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(start), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_tensor_intt_tensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).narrow(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(start), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_out_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* out, void* save_mean, void* save_invstd, void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(save_mean), from_raw::Tensor(save_invstd), from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_stats_tensor_double(void* input, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_stats(
        from_raw::Tensor(input), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double(void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::batch_norm_elemt(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_out_tensor_tensor_tensor_tensor_tensor_tensor_double(void* out, void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::batch_norm_elemt_out(
        from_raw::Tensor(out), from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* count)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats(
        from_raw::Tensor(input), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)count)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_tensor(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats_with_counts(
        from_raw::Tensor(input), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), from_raw::Tensor(counts))));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(void* grad_out, void* input, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_invstd, void* train, void* eps, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_backward(
        from_raw::Tensor(grad_out), from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)save_invstd)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* input_g, void* weight_g, void* bias_g)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_backward_reduce(
        from_raw::Tensor(grad_out), from_raw::Tensor(input), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<bool>*)input_g)->get(), ((LanternObject<bool>*)weight_g)->get(), ((LanternObject<bool>*)bias_g)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* mean_dy, void* mean_dy_xmu, void* count)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::batch_norm_backward_elemt(
        from_raw::Tensor(grad_out), from_raw::Tensor(input), from_raw::Tensor(mean), from_raw::Tensor(invstd), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), from_raw::Tensor(mean_dy), from_raw::Tensor(mean_dy_xmu), from_raw::Tensor(count)));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_update_stats_tensor_tensor_tensor_double(void* input, void* running_mean, void* running_var, void* momentum)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_update_stats(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::Tensor>>*)running_mean)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)running_var)->get(), ((LanternObject<double>*)momentum)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_is_vulkan_available()
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_vulkan_available(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_available()
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_nnpack_available(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_tensor_tensor_tensor_intarrayref_intarrayref(void* input, void* weight, void* bias, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_nnpack_spatial_convolution(
        from_raw::Tensor(input), from_raw::Tensor(weight), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool(void* input, void* grad_output, void* weight, void* padding, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_nnpack_spatial_convolution_backward(
        from_raw::Tensor(input), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(void* input, void* grad_output, void* weight, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_nnpack_spatial_convolution_backward_input(
        from_raw::Tensor(input), from_raw::Tensor(grad_output), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(void* input, void* weightsize, void* grad_output, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_nnpack_spatial_convolution_backward_weight(
        from_raw::Tensor(input), ((LanternObject<std::vector<int64_t>>*)weightsize)->get(), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ones(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ones(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ones_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ones_like(
        from_raw::Tensor(self), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pairwise_distance_tensor_tensor_double_double_bool(void* x1, void* x2, void* p, void* eps, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pairwise_distance(
        from_raw::Tensor(x1), from_raw::Tensor(x2), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cdist_tensor_tensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cdist(
        from_raw::Tensor(x1), from_raw::Tensor(x2), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__euclidean_dist_tensor_tensor(void* x1, void* x2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_euclidean_dist(
        from_raw::Tensor(x1), from_raw::Tensor(x2)));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_forward_tensor_tensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cdist_forward(
        from_raw::Tensor(x1), from_raw::Tensor(x2), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_backward_tensor_tensor_tensor_double_tensor(void* grad, void* x1, void* x2, void* p, void* cdist)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cdist_backward(
        from_raw::Tensor(grad), from_raw::Tensor(x1), from_raw::Tensor(x2), ((LanternObject<double>*)p)->get(), from_raw::Tensor(cdist)));
  LANTERN_FUNCTION_END
}

void* _lantern_pdist_tensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pdist(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_forward_tensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_pdist_forward(
        from_raw::Tensor(self), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_backward_tensor_tensor_double_tensor(void* grad, void* self, void* p, void* pdist)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_pdist_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self), ((LanternObject<double>*)p)->get(), from_raw::Tensor(pdist)));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_similarity_tensor_tensor_intt_double(void* x1, void* x2, void* dim, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cosine_similarity(
        from_raw::Tensor(x1), from_raw::Tensor(x2), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_permute_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::permute(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_permute_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).permute(
        ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::movedim(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).movedim(
        ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::movedim(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).movedim(
        ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_moveaxis_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::moveaxis(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_moveaxis_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).moveaxis(
        ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_moveaxis_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::moveaxis(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_moveaxis_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).moveaxis(
        ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_numpy_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).numpy_T(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pixel_shuffle_tensor_intt(void* self, void* upscale_factor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pixel_shuffle(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)upscale_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pixel_unshuffle_tensor_intt(void* self, void* downscale_factor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pixel_unshuffle(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)downscale_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_channel_shuffle_tensor_intt(void* self, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::channel_shuffle(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_pinned_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_pinned(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pin_memory_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pin_memory(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pinverse_tensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pinverse(
        from_raw::Tensor(self), ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pinverse_tensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pinverse(
        ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(void* input, void* target, void* log_input, void* full, void* eps, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::poisson_nll_loss(
        from_raw::Tensor(input), from_raw::Tensor(target), ((LanternObject<bool>*)log_input)->get(), ((LanternObject<bool>*)full)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rad2deg(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rad2deg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rad2deg_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rad2deg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rad2deg_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::deg2rad(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).deg2rad(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::deg2rad_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).deg2rad_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::deg2rad_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_scalar_tensor_scalar_tensoroptions(void* s, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scalar_tensor(
        ((LanternObject<torch::Scalar>*)s)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rand_like(
        from_raw::Tensor(self), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intarrayref_tensoroptions(void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intarrayref_generator_tensoroptions(void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_intarrayref_tensoroptions(void* low, void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_intarrayref_generator_tensoroptions(void* low, void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intarrayref(void* out, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intarrayref_generator(void* out, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intt_intarrayref(void* out, void* low, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intt_intarrayref_generator(void* out, void* low, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_tensor_intt_tensoroptions_memoryformat(void* self, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_like(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_tensor_intt_intt_tensoroptions_memoryformat(void* self, void* low, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randint_like(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randn_like(
        from_raw::Tensor(self), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_tensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_generator_tensoroptions(void* n, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_tensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randperm_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_tensor_intt_generator(void* out, void* n, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::randperm_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::range_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ravel_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ravel(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ravel_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ravel(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reciprocal(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).reciprocal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reciprocal_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).reciprocal_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reciprocal_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::neg(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).neg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::neg_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).neg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::neg_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::negative(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).negative(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::negative_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).negative_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::negative_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_tensor_intarrayref(void* self, void* repeats)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).repeat(
        ((LanternObject<std::vector<int64_t>>*)repeats)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor(void* repeats)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::repeat_interleave(
        from_raw::Tensor(repeats)));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor_tensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::repeat_interleave(
        from_raw::Tensor(self), from_raw::Tensor(repeats), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_tensor_tensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).repeat_interleave(
        from_raw::Tensor(repeats), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::repeat_interleave(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_tensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).repeat_interleave(
        ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reshape(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).reshape(
        ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_mkldnn_reshape(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).reshape_as(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_round_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::round(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).round(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::round_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).round_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::round_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu__tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_relu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::relu(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).relu(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_relu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::relu_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).relu_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_relu6_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::relu6(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_relu6__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::relu6_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_tensor_tensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prelu(
        from_raw::Tensor(self), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_tensor_tensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).prelu(
        from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_backward_tensor_tensor_tensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::prelu_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_backward_tensor_tensor_tensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(grad_output).prelu_backward(
        from_raw::Tensor(self), from_raw::Tensor(weight))));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gelu(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_backward_tensor_tensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gelu_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_infinitely_differentiable_gelu_backward_tensor_tensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::infinitely_differentiable_gelu_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardshrink(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).hardshrink(
        ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_backward_tensor_tensor_scalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardshrink_backward(
        from_raw::Tensor(grad_out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_backward_tensor_tensor_scalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(grad_out).hardshrink_backward(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rsqrt(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rsqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rsqrt_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rsqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rsqrt_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_select_tensor_dimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::select(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_tensor_dimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).select(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_tensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::select(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_tensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).select(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_backward_tensor_intarrayref_intt_intt(void* grad, void* input_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::select_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_selu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::selu(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_selu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::selu_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_celu_tensor_scalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::celu(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_celu__tensor_scalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::celu_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::silu(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_silu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::silu_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::silu_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::silu_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mish(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_mish__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mish_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mish_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mish_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sigmoid(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sigmoid(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sigmoid_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sigmoid_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sigmoid_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logit(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit_tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logit(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit__tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logit_(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit__tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).logit_(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_out_tensor_tensor_double(void* out, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logit_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sin(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sin_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sin_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinc(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sinc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinc_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sinc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinc_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sinh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_detach_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::detach(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).detach(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_detach__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::detach_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).detach_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_size_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_size_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_size_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).size(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_tensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slice(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)start)->get(), ((LanternObject<c10::optional<int64_t>>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slice_tensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).slice(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)start)->get(), ((LanternObject<c10::optional<int64_t>>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_backward_tensor_intarrayref_intt_intt_intt_intt(void* grad, void* input_sizes, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slice_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slogdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slogdet(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slogdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).slogdet(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_smm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::smm(
        from_raw::Tensor(self), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_smm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).smm(
        from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softmax(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).softmax(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_softmax_backward_data(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_split(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).unsafe_split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::split(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_split_with_sizes(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).unsafe_split_with_sizes(
        ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::split_with_sizes(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).split_with_sizes(
        ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::hsplit(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).hsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::hsplit(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).hsplit(
        ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::vsplit(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).vsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::vsplit(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).vsplit(
        ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::dsplit(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dsplit_tensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).dsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::dsplit(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dsplit_tensor_intarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).dsplit(
        ((LanternObject<std::vector<int64_t>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::squeeze(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::squeeze(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::squeeze(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).squeeze_(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sspaddmm(
        from_raw::Tensor(self), from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sspaddmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sspaddmm(
        from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sspaddmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::stack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::stack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__stack_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_stack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__stack_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_stack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hstack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::vstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::vstack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dstack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::stft(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)window)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).stft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)window)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::istft(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)window)->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).istft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)window)->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).stride(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).stride(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sum(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sum(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sum(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sum(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sum(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sum(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nansum(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nansum(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nansum(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nansum(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nansum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_to_size_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sum_to_size(
        ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sqrt(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sqrt_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sqrt_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_square_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::square(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).square(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_square__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::square_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).square_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_square_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::square_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std(
        from_raw::Tensor(self), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).std(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).std(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).std(
        ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        from_raw::Tensor(self), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_intarrayref_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).std(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).std(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_dimnamelist_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::std_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prod(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).prod(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_intt_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prod(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_intt_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).prod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_tensor_tensor_intt_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_dimname_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prod(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_dimname_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).prod(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_tensor_tensor_dimname_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::prod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::t(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).t(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).t_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tan(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tan_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tan_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tanh(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tanh_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tanh_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_tensordot_tensor_tensor_intarrayref_intarrayref(void* self, void* other, void* dims_self, void* dims_other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tensordot(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<std::vector<int64_t>>*)dims_self)->get(), ((LanternObject<std::vector<int64_t>>*)dims_other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensordot_out_tensor_tensor_tensor_intarrayref_intarrayref(void* out, void* self, void* other, void* dims_self, void* dims_other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tensordot_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<std::vector<int64_t>>*)dims_self)->get(), ((LanternObject<std::vector<int64_t>>*)dims_other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_tensor_scalar_scalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::threshold(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold__tensor_scalar_scalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::threshold_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_out_tensor_tensor_scalar_scalar(void* out, void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::threshold_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_backward_out_tensor_tensor_tensor_scalar(void* grad_input, void* grad_output, void* self, void* threshold)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::threshold_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_backward_tensor_tensor_scalar(void* grad_output, void* self, void* threshold)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::threshold_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tile_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tile(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tile_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tile(
        ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::transpose(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).transpose(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_tensor_dimname_dimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::transpose(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim0)->get(), ((LanternObject<torch::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_tensor_dimname_dimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).transpose(
        ((LanternObject<torch::Dimname>*)dim0)->get(), ((LanternObject<torch::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_mkldnn_transpose(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).transpose_(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_mkldnn_transpose_(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_one_hot_tensor_intt(void* self, void* num_classes)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::one_hot(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)num_classes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flip_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flip(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flip_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flip(
        ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fliplr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fliplr(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fliplr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fliplr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_flipud_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flipud(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flipud_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).flipud(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_roll_tensor_intarrayref_intarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::roll(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)shifts)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_roll_tensor_intarrayref_intarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).roll(
        ((LanternObject<std::vector<int64_t>>*)shifts)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rot90_tensor_intt_intarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rot90(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rot90_tensor_intt_intarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).rot90(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_tensor_tensor_intt(void* y, void* x, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trapz(
        from_raw::Tensor(y), from_raw::Tensor(x), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_tensor_double_intt(void* y, void* dx, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trapz(
        from_raw::Tensor(y), ((LanternObject<double>*)dx)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_trilinear(
        from_raw::Tensor(i1), from_raw::Tensor(i2), from_raw::Tensor(i3), ((LanternObject<std::vector<int64_t>>*)expand1)->get(), ((LanternObject<std::vector<int64_t>>*)expand2)->get(), ((LanternObject<std::vector<int64_t>>*)expand3)->get(), ((LanternObject<std::vector<int64_t>>*)sumdim)->get(), ((LanternObject<int64_t>*)unroll_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::triplet_margin_loss(
        from_raw::Tensor(anchor), from_raw::Tensor(positive), from_raw::Tensor(negative), ((LanternObject<double>*)margin)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)swap)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trunc(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).trunc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trunc_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).trunc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trunc_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fix(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fix(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fix_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fix_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fix_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_type_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).type_as(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern__has_compatible_shallow_copy_type_tensor_tensor(void* self, void* from)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_has_compatible_shallow_copy_type(
        from_raw::Tensor(self), from_raw::Tensor(from)));
  LANTERN_FUNCTION_END
}

void* _lantern__unique_tensor_bool_bool(void* self, void* sorted, void* return_inverse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique(
        from_raw::Tensor(self), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_tensor_intt_bool_bool_bool(void* self, void* dim, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_consecutive_tensor_bool_bool_intt(void* self, void* return_inverse, void* return_counts, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_consecutive(
        from_raw::Tensor(self), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_consecutive_tensor_intt_bool_bool(void* self, void* dim, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim_consecutive(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unique2_tensor_bool_bool_bool(void* self, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique2(
        from_raw::Tensor(self), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unsafe_view_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_unsafe_view(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsqueeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::unsqueeze(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).unsqueeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze__tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).unsqueeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vander_tensor_intt_bool(void* x, void* N, void* increasing)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::vander(
        from_raw::Tensor(x), ((LanternObject<c10::optional<int64_t>>*)N)->get(), ((LanternObject<bool>*)increasing)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var(
        from_raw::Tensor(self), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).var(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).var(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).var(
        ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_intarrayref_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).var(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).var(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_dimnamelist_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::var_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        from_raw::Tensor(self), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_intarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_dimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        from_raw::Tensor(self), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).view_as(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::where(
        from_raw::Tensor(condition), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(condition).where(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_scalar_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::where(
        from_raw::Tensor(condition), ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_tensor_scalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::where(
        from_raw::Tensor(condition), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_scalar_scalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::where(
        from_raw::Tensor(condition), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor(void* condition)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::where(
        from_raw::Tensor(condition)));
  LANTERN_FUNCTION_END
}

void* _lantern__s_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_s_where(
        from_raw::Tensor(condition), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_except_dim_tensor_intt_intt(void* v, void* pow, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm_except_dim(
        from_raw::Tensor(v), ((LanternObject<int64_t>*)pow)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_tensor_tensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_weight_norm(
        from_raw::Tensor(v), from_raw::Tensor(g), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_tensor_tensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface(
        from_raw::Tensor(v), from_raw::Tensor(g), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface_backward(
        from_raw::Tensor(grad_w), from_raw::Tensor(saved_v), from_raw::Tensor(saved_g), from_raw::Tensor(saved_norms), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_differentiable_backward(
        from_raw::Tensor(grad_w), from_raw::Tensor(saved_v), from_raw::Tensor(saved_g), from_raw::Tensor(saved_norms), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::zeros_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::zeros_like(
        from_raw::Tensor(self), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_grad_tensor_tensor(void* self, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_standard_gamma_grad(
        from_raw::Tensor(self), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_standard_gamma(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dirichlet_grad_tensor_tensor_tensor(void* x, void* alpha, void* total)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_dirichlet_grad(
        from_raw::Tensor(x), from_raw::Tensor(alpha), from_raw::Tensor(total)));
  LANTERN_FUNCTION_END
}

void* _lantern__sample_dirichlet_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sample_dirichlet(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::poisson(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binomial_tensor_tensor_generator(void* count, void* prob, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::binomial(
        from_raw::Tensor(count), from_raw::Tensor(prob), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::native_norm(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::native_norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sum(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sum(
        from_raw::Tensor(self), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sum(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_intarrayref_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sum(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_backward_tensor_tensor_intarrayref(void* grad, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_sum_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_softmax(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_softmax_backward_data(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_log_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_log_softmax(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_log_softmax(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_log_softmax_backward_data(
        from_raw::Tensor(grad_output), from_raw::Tensor(output), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_scalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_scalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_intarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_intarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_dimnamelist_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_dimnamelist_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_dimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_dimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frexp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::frexp(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frexp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).frexp(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_frexp_out_tensor_tensor_tensor(void* mantissa, void* exponent, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::frexp_out(
        from_raw::Tensor(mantissa), from_raw::Tensor(exponent), from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frobenius_norm(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frobenius_norm(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::frobenius_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_tensor_bool(void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nuclear_norm(
        from_raw::Tensor(self), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_tensor_tensor_bool(void* out, void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nuclear_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nuclear_norm(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nuclear_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clone_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::clone(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clone_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).clone(
        ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_positive_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::positive(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_positive_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).positive(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_resize_as__tensor_tensor_memoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::resize_as_(
        from_raw::Tensor(self), from_raw::Tensor(the_template), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize_as__tensor_tensor_memoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).resize_as_(
        from_raw::Tensor(the_template), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_resize_as_sparse__tensor_tensor(void* self, void* the_template)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::resize_as_sparse_(
        from_raw::Tensor(self), from_raw::Tensor(the_template)));
  LANTERN_FUNCTION_END
}

void* _lantern_zero__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::zero_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_zero__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).zero_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sub_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sub(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sub(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sub_(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sub(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sub(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sub_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::subtract_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::subtract(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).subtract(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).subtract_(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::subtract(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).subtract(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).subtract_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rsub(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_out_tensor_tensor_tensor(void* out, void* self, void* values)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::heaviside_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(values)));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::heaviside(
        from_raw::Tensor(self), from_raw::Tensor(values)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside_tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).heaviside(
        from_raw::Tensor(values)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside__tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).heaviside_(
        from_raw::Tensor(values)));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rsub(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* sparse, void* dense, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_addmm(
        from_raw::Tensor(self), from_raw::Tensor(sparse), from_raw::Tensor(dense), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addmm(
        from_raw::Tensor(self), from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addmm(
        from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm__tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addmm_(
        from_raw::Tensor(mat1), from_raw::Tensor(mat2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_csr_tensor_tensor_tensor_tensor_intarrayref_tensoroptions(void* crow_indices, void* col_indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_csr_tensor(
        from_raw::Tensor(crow_indices), from_raw::Tensor(col_indices), from_raw::Tensor(values), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_csr_tensor_tensor_tensor_tensor_tensoroptions(void* crow_indices, void* col_indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_csr_tensor(
        from_raw::Tensor(crow_indices), from_raw::Tensor(col_indices), from_raw::Tensor(values), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sparse_coo_tensor(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sparse_coo_tensor(
        from_raw::Tensor(indices), from_raw::Tensor(values), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sparse_coo_tensor(
        from_raw::Tensor(indices), from_raw::Tensor(values), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_coo_tensor_unsafe(
        from_raw::Tensor(indices), from_raw::Tensor(values), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__validate_sparse_coo_tensor_args_tensor_tensor_intarrayref(void* indices, void* values, void* size)
{
  LANTERN_FUNCTION_START
    torch::_validate_sparse_coo_tensor_args(from_raw::Tensor(indices), from_raw::Tensor(values), ((LanternObject<std::vector<int64_t>>*)size)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_coo_tensor_with_dims(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_sparse_coo_tensor_with_dims_and_tensors(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), from_raw::Tensor(indices), from_raw::Tensor(values), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sparse_resize_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize_and_clear__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sparse_resize_and_clear_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_mask_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sparse_mask(
        from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_dense_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to_dense(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_to_dense_backward_tensor_tensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::to_dense_backward(
        from_raw::Tensor(grad), from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_dim_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).sparse_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimi_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self)._dimI(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dense_dim_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).dense_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self)._dimV(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__nnz_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self)._nnz(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_coalesce_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).coalesce(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__coalesce_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_coalesce(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_coalesced_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_coalesced(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self)._indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__values_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self)._values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__coalesced__tensor_bool(void* self, void* coalesced)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self)._coalesced_(
        ((LanternObject<bool>*)coalesced)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_values_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_crow_indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).crow_indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_col_indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).col_indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_out_tensor_tensor_tensor(void* out, void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hspmm_out(
        from_raw::Tensor(out), from_raw::Tensor(mat1), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_tensor_tensor(void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hspmm(
        from_raw::Tensor(mat1), from_raw::Tensor(mat2)));
  LANTERN_FUNCTION_END
}

void* _lantern_copy_sparse_to_sparse__tensor_tensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::copy_sparse_to_sparse_(
        from_raw::Tensor(self), from_raw::Tensor(src), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unbind(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).unbind(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unbind(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).unbind(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_tensor_intt(void* self, void* sparse_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to_sparse(
        ((LanternObject<int64_t>*)sparse_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to_sparse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_mkldnn_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to_mkldnn(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_reorder_conv2d_weight(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv3d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_reorder_conv3d_weight(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_to_mkldnn_backward_tensor_tensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::to_mkldnn_backward(
        from_raw::Tensor(grad), from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_tensor_double_intt_scalartype(void* self, void* scale, void* zero_point, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantize_per_tensor(
        from_raw::Tensor(self), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_tensorlist_tensor_tensor_scalartype(void* tensors, void* scales, void* zero_points, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::quantize_per_tensor(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), from_raw::Tensor(scales), from_raw::Tensor(zero_points), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype(void* self, void* scales, void* zero_points, void* axis, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantize_per_channel(
        from_raw::Tensor(self), from_raw::Tensor(scales), from_raw::Tensor(zero_points), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dequantize_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dequantize(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dequantize_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).dequantize(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_dequantize_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::dequantize(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_q_scale_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<double>(torch::q_scale(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_scale_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<double>(from_raw::Tensor(self).q_scale(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_zero_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_zero_point(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_zero_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).q_zero_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_scales_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::q_per_channel_scales(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_scales_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).q_per_channel_scales(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_zero_points_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::q_per_channel_zero_points(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_zero_points_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).q_per_channel_zero_points(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_axis_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_per_channel_axis(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_axis_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(from_raw::Tensor(self).q_per_channel_axis(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_int_repr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::int_repr(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_int_repr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).int_repr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_tensor_quantized_tensor_tensor_double_intt(void* self, void* scale, void* zero_point)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_make_per_tensor_quantized_tensor(
        from_raw::Tensor(self), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt(void* self, void* scale, void* zero_point, void* axis)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_make_per_channel_quantized_tensor(
        from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)axis)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qscheme_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::QScheme>(from_raw::Tensor(self).qscheme(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fake_quantize_per_tensor_affine(
        from_raw::Tensor(self), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_cachemask_tensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fake_quantize_per_tensor_affine_cachemask(
        from_raw::Tensor(self), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_cachemask_backward_tensor_tensor(void* grad, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fake_quantize_per_tensor_affine_cachemask_backward(
        from_raw::Tensor(grad), from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_tensor_tensor_tensor_intt_intt_double(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fake_quantize_learnable_per_tensor_affine(
        from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_backward_tensor_tensor_tensor_tensor_intt_intt_double(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_tensor_affine_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fake_quantize_per_channel_affine(
        from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_cachemask_tensor_tensor_tensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fake_quantize_per_channel_affine_cachemask(
        from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_cachemask_backward_tensor_tensor(void* grad, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fake_quantize_per_channel_affine_cachemask_backward(
        from_raw::Tensor(grad), from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_tensor_tensor_tensor_intt_intt_intt_double(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_fake_quantize_learnable_per_channel_affine(
        from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt_double(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_channel_affine_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self), from_raw::Tensor(scale), from_raw::Tensor(zero_point), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__choose_qparams_per_tensor_tensor_bool(void* self, void* reduce_range)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_choose_qparams_per_tensor(
        from_raw::Tensor(self), ((LanternObject<bool>*)reduce_range)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__saturate_weight_to_fp16_tensor(void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_saturate_weight_to_fp16(
        from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_choose_qparams_optimized_tensor_intt_intt_double_intt(void* input, void* numel, void* n_bins, void* ratio, void* bit_width)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::choose_qparams_optimized(
        from_raw::Tensor(input), ((LanternObject<int64_t>*)numel)->get(), ((LanternObject<int64_t>*)n_bins)->get(), ((LanternObject<double>*)ratio)->get(), ((LanternObject<int64_t>*)bit_width)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_tensoroptions_bool_bool_memoryformat(void* self, void* options, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to(
        ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_device_scalartype_bool_bool_memoryformat(void* self, void* device, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to(
        ((LanternPtr<torch::Device>*)device)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_scalartype_bool_bool_memoryformat(void* self, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to(
        ((LanternObject<torch::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_tensor_bool_bool_memoryformat(void* self, void* other, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).to(
        from_raw::Tensor(other), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_meshgrid_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::meshgrid(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cartesian_prod_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cartesian_prod(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_combinations_tensor_intt_bool(void* self, void* r, void* with_replacement)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::combinations(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)r)->get(), ((LanternObject<bool>*)with_replacement)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_item_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Scalar>(from_raw::Tensor(self).item(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_tensor_tensor(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        from_raw::Tensor(tensor), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_tensor_scalar(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        from_raw::Tensor(tensor), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_scalar_tensor(void* scalar, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Scalar>*)scalar)->get(), from_raw::Tensor(tensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_scalar_scalar(void* scalar1, void* scalar2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Scalar>*)scalar1)->get(), ((LanternObject<torch::Scalar>*)scalar2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_can_cast_scalartype_scalartype(void* from, void* to)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::can_cast(
        ((LanternObject<torch::ScalarType>*)from)->get(), ((LanternObject<torch::ScalarType>*)to)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_promote_types_scalartype_scalartype(void* type1, void* type2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::promote_types(
        ((LanternObject<torch::ScalarType>*)type1)->get(), ((LanternObject<torch::ScalarType>*)type2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__local_scalar_dense_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Scalar>(torch::_local_scalar_dense(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* cx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell(
        from_raw::Tensor(input_gates), from_raw::Tensor(hidden_gates), from_raw::Tensor(cx), ((LanternObject<c10::optional<torch::Tensor>>*)input_bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)hidden_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool(void* grad_hy, void* grad_cy, void* cx, void* cy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)grad_hy)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_cy)->get(), from_raw::Tensor(cx), from_raw::Tensor(cy), from_raw::Tensor(workspace), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* grad_cy, void* input_gates, void* hidden_gates, void* input_bias, void* hidden_bias, void* cx, void* cy)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_lstm_cell_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)grad_hy)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)grad_cy)->get(), from_raw::Tensor(input_gates), from_raw::Tensor(hidden_gates), ((LanternObject<c10::optional<torch::Tensor>>*)input_bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)hidden_bias)->get(), from_raw::Tensor(cx), from_raw::Tensor(cy))));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell(
        from_raw::Tensor(input_gates), from_raw::Tensor(hidden_gates), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)input_bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)hidden_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool(void* grad_hy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell_backward(
        from_raw::Tensor(grad_hy), from_raw::Tensor(workspace), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_gru_cell_backward(
        from_raw::Tensor(grad_hy), from_raw::Tensor(input_gates), from_raw::Tensor(hidden_gates), from_raw::Tensor(hx), ((LanternObject<c10::optional<torch::Tensor>>*)input_bias)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)hidden_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        from_raw::Tensor(data), from_raw::Tensor(batch_sizes), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        from_raw::Tensor(input), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        from_raw::Tensor(data), from_raw::Tensor(batch_sizes), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        from_raw::Tensor(input), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        from_raw::Tensor(data), from_raw::Tensor(batch_sizes), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        from_raw::Tensor(input), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        from_raw::Tensor(data), from_raw::Tensor(batch_sizes), from_raw::Tensor(hx), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm_cell(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), ((LanternObject<c10::optional<torch::Tensor>>*)b_ih)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)b_hh)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gru_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), ((LanternObject<c10::optional<torch::Tensor>>*)b_ih)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)b_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rnn_tanh_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), ((LanternObject<c10::optional<torch::Tensor>>*)b_ih)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)b_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rnn_relu_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), ((LanternObject<c10::optional<torch::Tensor>>*)b_ih)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)b_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::quantized_lstm_cell(
        from_raw::Tensor(input), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), from_raw::Tensor(b_ih), from_raw::Tensor(b_hh), from_raw::Tensor(packed_ih), from_raw::Tensor(packed_hh), from_raw::Tensor(col_offsets_ih), from_raw::Tensor(col_offsets_hh), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_gru_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), from_raw::Tensor(b_ih), from_raw::Tensor(b_hh), from_raw::Tensor(packed_ih), from_raw::Tensor(packed_hh), from_raw::Tensor(col_offsets_ih), from_raw::Tensor(col_offsets_hh), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_rnn_relu_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), from_raw::Tensor(b_ih), from_raw::Tensor(b_hh), from_raw::Tensor(packed_ih), from_raw::Tensor(packed_hh), from_raw::Tensor(col_offsets_ih), from_raw::Tensor(col_offsets_hh), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantized_rnn_tanh_cell(
        from_raw::Tensor(input), from_raw::Tensor(hx), from_raw::Tensor(w_ih), from_raw::Tensor(w_hh), from_raw::Tensor(b_ih), from_raw::Tensor(b_hh), from_raw::Tensor(packed_ih), from_raw::Tensor(packed_hh), from_raw::Tensor(col_offsets_ih), from_raw::Tensor(col_offsets_hh), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_tensor_tensor_bool(void* input, void* lengths, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pack_padded_sequence(
        from_raw::Tensor(input), from_raw::Tensor(lengths), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(void* grad, void* input_size, void* batch_sizes, void* batch_first)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_pack_padded_sequence_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), from_raw::Tensor(batch_sizes), ((LanternObject<bool>*)batch_first)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt(void* data, void* batch_sizes, void* batch_first, void* padding_value, void* total_length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pad_packed_sequence(
        from_raw::Tensor(data), from_raw::Tensor(batch_sizes), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<torch::Scalar>*)padding_value)->get(), ((LanternObject<int64_t>*)total_length)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_storage(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).set_(
        ((LanternObject<torch::Storage>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_storage_intt_intarrayref_intarrayref(void* self, void* source, void* storage_offset, void* size, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).set_(
        ((LanternObject<torch::Storage>*)source)->get(), ((LanternObject<int64_t>*)storage_offset)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_tensor(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).set_(
        from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).set_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_set_to_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).is_set_to(
        from_raw::Tensor(tensor)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_fill_(
        from_raw::Tensor(mask), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_fill(
        from_raw::Tensor(self), from_raw::Tensor(mask), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_fill(
        from_raw::Tensor(mask), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_fill_(
        from_raw::Tensor(mask), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_fill(
        from_raw::Tensor(self), from_raw::Tensor(mask), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_fill(
        from_raw::Tensor(mask), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter__tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_scatter_(
        from_raw::Tensor(mask), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_scatter_tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_scatter(
        from_raw::Tensor(self), from_raw::Tensor(mask), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter_tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_scatter(
        from_raw::Tensor(mask), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).view(
        ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).view(
        ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_put__tensor_tensor_tensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).put_(
        from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_put_tensor_tensor_tensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::put(
        from_raw::Tensor(self), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_put_tensor_tensor_tensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).put(
        from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_add_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add__tensor_intt_tensor_tensor_scalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_add_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_add(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_add(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_tensor_intt_tensor_tensor_scalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_add(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_tensor_intt_tensor_tensor_scalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_add(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_tensor_dimname_tensor_tensor_scalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_add(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_tensor_dimname_tensor_tensor_scalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_add(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_fill(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_fill(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill_(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill_(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_fill(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_fill(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_fill(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(value)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_tensor_stdstring(void* self, void* dim, void* index, void* src, void* reduce)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_scalar_stdstring(void* self, void* dim, void* index, void* value, void* reduce)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<torch::Scalar>*)value)->get(), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_add_(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter_add(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_add(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::scatter_add(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).scatter_add(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(src)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).eq_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).eq_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_and_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_and_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_and(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_and(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_and(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_and(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_and_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_and_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern___and___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__and__(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__and__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___and___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__and__(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__and__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__iand__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__iand__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_or_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_or_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_or(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_or(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_or(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_or(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_or_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_or_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern___or___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__or__(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__or__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___or___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__or__(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__or__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ior__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ior__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_xor_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_xor_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_xor(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_xor(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bitwise_xor(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_xor(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_xor_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).bitwise_xor_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__xor__(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__xor__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__xor__(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__xor__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ixor__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ixor__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__lshift__(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__lshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__lshift__(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__lshift__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ilshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__ilshift__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__rshift__(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__rshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::__rshift__(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__rshift__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__irshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).__irshift__(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril__tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tril_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu__tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).triu_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).digamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm__tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).renorm_(
        ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lerp_(
        from_raw::Tensor(end), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lerp_(
        from_raw::Tensor(end), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmod_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmod_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).remainder_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).remainder_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addbmm_(
        from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addbmm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addbmm(
        from_raw::Tensor(self), from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addbmm(
        from_raw::Tensor(batch1), from_raw::Tensor(batch2), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addcdiv_(
        from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_intt_intt_generator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).random_(
        ((LanternObject<int64_t>*)from)->get(), ((LanternObject<c10::optional<int64_t>>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_intt_generator(void* self, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).random_(
        ((LanternObject<int64_t>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).random_(
        ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_uniform__tensor_double_double_generator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).uniform_(
        ((LanternObject<double>*)from)->get(), ((LanternObject<double>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cauchy__tensor_double_double_generator(void* self, void* median, void* sigma, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cauchy_(
        ((LanternObject<double>*)median)->get(), ((LanternObject<double>*)sigma)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).log_normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exponential__tensor_double_generator(void* self, void* lambd, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).exponential_(
        ((LanternObject<double>*)lambd)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geometric__tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).geometric_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diag_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diag(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).diag(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_backward_tensor_intarrayref_intt(void* grad, void* input_sizes, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::diag_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_out_tensor_tensor_tensor_intt(void* out, void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cross_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_tensor_tensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cross(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cross_tensor_tensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cross(
        from_raw::Tensor(other), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::triu_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::triu(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).triu(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tril_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tril(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).tril(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tril_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::triu_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trace(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trace_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).trace(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_backward_tensor_intarrayref(void* grad, void* sizes)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::trace_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ne_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ne(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ne(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ne_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ne(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ne(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ne_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ne_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::not_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::not_equal(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).not_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::not_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::not_equal(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).not_equal(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).not_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).not_equal_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eq_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eq(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).eq(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eq_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::eq(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).eq(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ge_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ge(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ge(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ge_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ge(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ge(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ge_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ge_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_equal(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_equal(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_equal(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_equal_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::le_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::le(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).le(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::le_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_le_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::le(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).le(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).le_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).le_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_equal(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_equal_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_equal(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_equal(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_equal_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gt_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gt(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gt(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gt_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gt(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gt(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gt_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gt_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::greater(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).greater_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lt_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lt(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lt(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lt_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lt(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lt(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lt_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lt_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_less_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::less(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).less_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_take_out_tensor_tensor_tensor(void* out, void* self, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::take_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_take_tensor_tensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::take(
        from_raw::Tensor(self), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_take_tensor_tensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).take(
        from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_take_along_dim_out_tensor_tensor_tensor_intt(void* out, void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::take_along_dim_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_along_dim_tensor_tensor_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::take_along_dim(
        from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_take_along_dim_tensor_tensor_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).take_along_dim(
        from_raw::Tensor(indices), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_tensor_tensor_intt_tensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_select_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_tensor_intt_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_select(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_tensor_intt_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_select(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_tensor_tensor_dimname_tensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_select_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_tensor_dimname_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_select(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_tensor_dimname_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).index_select(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_backward_tensor_intarrayref_intt_tensor(void* grad, void* self_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::index_select_backward(
        from_raw::Tensor(grad), ((LanternObject<std::vector<int64_t>>*)self_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index)));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_out_tensor_tensor_tensor(void* out, void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_select_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_select(
        from_raw::Tensor(self), from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_select_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).masked_select(
        from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_backward_tensor_tensor_tensor(void* grad, void* input, void* mask)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::masked_select_backward(
        from_raw::Tensor(grad), from_raw::Tensor(input), from_raw::Tensor(mask)));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nonzero_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nonzero(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_numpy_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::nonzero_numpy(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_numpy_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(from_raw::Tensor(self).nonzero_numpy(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_tensor_tensor_intt_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gather_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_tensor_intt_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gather(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_tensor_intt_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gather(
        ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_backward_tensor_tensor_intt_tensor_bool(void* grad, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gather_backward(
        from_raw::Tensor(grad), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_tensor_tensor_dimname_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gather_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_tensor_dimname_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::gather(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_tensor_dimname_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).gather(
        ((LanternObject<torch::Dimname>*)dim)->get(), from_raw::Tensor(index), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__gather_sparse_backward_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* grad)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_gather_sparse_backward(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(grad)));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addcmul_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addcmul(
        from_raw::Tensor(self), from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addcmul(
        from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addcmul_(
        from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addcdiv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::addcdiv(
        from_raw::Tensor(self), from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).addcdiv(
        from_raw::Tensor(tensor1), from_raw::Tensor(tensor2), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_entropy_loss_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cross_entropy_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_out_tensor_tensor_tensor_tensor(void* X, void* qr, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq_out(
        from_raw::Tensor(X), from_raw::Tensor(qr), from_raw::Tensor(self), from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq(
        from_raw::Tensor(self), from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lstsq_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).lstsq(
        from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool(void* X, void* M, void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve_out(
        from_raw::Tensor(X), from_raw::Tensor(M), from_raw::Tensor(self), from_raw::Tensor(A), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve(
        from_raw::Tensor(self), from_raw::Tensor(A), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triangular_solve_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).triangular_solve(
        from_raw::Tensor(A), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_out_tensor_tensor_tensor_bool_bool(void* e, void* V, void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig_out(
        from_raw::Tensor(e), from_raw::Tensor(V), from_raw::Tensor(self), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig(
        from_raw::Tensor(self), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_symeig_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).symeig(
        ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__symeig_helper_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_symeig_helper(
        from_raw::Tensor(self), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_out_tensor_tensor_tensor_bool(void* e, void* v, void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig_out(
        from_raw::Tensor(e), from_raw::Tensor(v), from_raw::Tensor(self), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_tensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig(
        from_raw::Tensor(self), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eig_tensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).eig(
        ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool(void* U, void* S, void* V, void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd_out(
        from_raw::Tensor(U), from_raw::Tensor(S), from_raw::Tensor(V), from_raw::Tensor(self), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd(
        from_raw::Tensor(self), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_svd_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).svd(
        ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__svd_helper_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_svd_helper(
        from_raw::Tensor(self), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_swapaxes_tensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::swapaxes(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapaxes_tensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).swapaxes(
        ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapaxes__tensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).swapaxes_(
        ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_swapdims_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::swapdims(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapdims_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).swapdims(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapdims__tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).swapdims_(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_out_tensor_tensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky(
        from_raw::Tensor(self), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cholesky(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_out_tensor_tensor_tensor_bool(void* out, void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky_solve_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(input2), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_tensor_tensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky_solve(
        from_raw::Tensor(self), from_raw::Tensor(input2), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_solve_tensor_tensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cholesky_solve(
        from_raw::Tensor(input2), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cholesky_solve_helper_tensor_tensor_bool(void* self, void* A, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cholesky_solve_helper(
        from_raw::Tensor(self), from_raw::Tensor(A), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve(
        from_raw::Tensor(self), from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_solve_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).solve(
        from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_out_tensor_tensor_tensor_tensor(void* solution, void* lu, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve_out(
        from_raw::Tensor(solution), from_raw::Tensor(lu), from_raw::Tensor(self), from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern__solve_helper_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_solve_helper(
        from_raw::Tensor(self), from_raw::Tensor(A))));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky_inverse(
        from_raw::Tensor(self), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_inverse_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).cholesky_inverse(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_out_tensor_tensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::cholesky_inverse_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_out_tensor_tensor_tensor_bool(void* Q, void* R, void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr_out(
        from_raw::Tensor(Q), from_raw::Tensor(R), from_raw::Tensor(self), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_tensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr(
        from_raw::Tensor(self), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qr_tensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).qr(
        ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_out_tensor_tensor_tensor(void* a, void* tau, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf_out(
        from_raw::Tensor(a), from_raw::Tensor(tau), from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geqrf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).geqrf(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_tensor_tensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::orgqr(
        from_raw::Tensor(self), from_raw::Tensor(input2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_orgqr_tensor_tensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).orgqr(
        from_raw::Tensor(input2)));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_out_tensor_tensor_tensor(void* out, void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::orgqr_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(input2)));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool(void* out, void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ormqr_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(input2), from_raw::Tensor(input3), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_tensor_tensor_tensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ormqr(
        from_raw::Tensor(self), from_raw::Tensor(input2), from_raw::Tensor(input3), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ormqr_tensor_tensor_tensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ormqr(
        from_raw::Tensor(input2), from_raw::Tensor(input3), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__lu_with_info_tensor_bool_bool(void* self, void* pivot, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_lu_with_info(
        from_raw::Tensor(self), ((LanternObject<bool>*)pivot)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_out_tensor_tensor_tensor_tensor(void* out, void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lu_solve_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(LU_data), from_raw::Tensor(LU_pivots)));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lu_solve(
        from_raw::Tensor(self), from_raw::Tensor(LU_data), from_raw::Tensor(LU_pivots)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lu_solve_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lu_solve(
        from_raw::Tensor(LU_data), from_raw::Tensor(LU_pivots)));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_unpack_tensor_tensor_bool_bool(void* LU_data, void* LU_pivots, void* unpack_data, void* unpack_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lu_unpack(
        from_raw::Tensor(LU_data), from_raw::Tensor(LU_pivots), ((LanternObject<bool>*)unpack_data)->get(), ((LanternObject<bool>*)unpack_pivots)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_unpack_out_tensor_tensor_tensor_tensor_tensor_bool_bool(void* P, void* L, void* U, void* LU_data, void* LU_pivots, void* unpack_data, void* unpack_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lu_unpack_out(
        from_raw::Tensor(P), from_raw::Tensor(L), from_raw::Tensor(U), from_raw::Tensor(LU_data), from_raw::Tensor(LU_pivots), ((LanternObject<bool>*)unpack_data)->get(), ((LanternObject<bool>*)unpack_pivots)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_out_tensor_tensor_intt_bool_generator(void* out, void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multinomial_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_tensor_intt_bool_generator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multinomial(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multinomial_tensor_intt_bool_generator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).multinomial(
        ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lgamma_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lgamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lgamma(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lgamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::digamma_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::digamma(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).digamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_polygamma_out_tensor_intt_tensor(void* out, void* n, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::polygamma_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get(), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_polygamma__tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).polygamma_(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erfinv(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erfinv(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).erfinv_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::erfinv_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::i0(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).i0(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::i0_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).i0_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::i0_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sign(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sign(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).sign_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sign_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::signbit(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_signbit_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).signbit(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::signbit_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_dist_tensor_tensor_scalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::dist(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dist_tensor_tensor_scalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).dist(
        from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atan2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atan2_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::atan2(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).atan2(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_tensor_tensor_tensor_scalar(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lerp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(end), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_tensor_tensor_tensor_tensor(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lerp_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(end), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lerp(
        from_raw::Tensor(self), from_raw::Tensor(end), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lerp(
        from_raw::Tensor(end), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::lerp(
        from_raw::Tensor(self), from_raw::Tensor(end), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).lerp(
        from_raw::Tensor(end), from_raw::Tensor(weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_out_tensor_tensor_intt_scalar_scalar(void* out, void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::histc_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_tensor_intt_scalar_scalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::histc(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_histc_tensor_intt_scalar_scalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).histc(
        ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmod(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmod(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmod(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmod(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hypot_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hypot(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).hypot(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).hypot_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_igamma_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::igamma_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_igamma_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::igamma(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igamma_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).igamma(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igamma__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).igamma_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_igammac_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::igammac_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_igammac_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::igammac(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igammac_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).igammac(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igammac__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).igammac_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nextafter_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nextafter(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nextafter(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nextafter_(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::remainder_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::remainder(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).remainder(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::remainder_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::remainder(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).remainder(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::min(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).min(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fmin_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmin(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmin_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmin(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_fmin_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmin_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).max(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fmax_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmax(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmax_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).fmax(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_fmax_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fmax_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::maximum(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_maximum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).maximum(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::maximum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).max(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::minimum(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_minimum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).minimum(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::minimum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::min_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::min(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).min(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile(
        from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).quantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_tensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile(
        from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).quantile(
        from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile(
        from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nanquantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_tensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile(
        from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nanquantile(
        from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_double_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile(
        from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).quantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_tensor_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_tensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::quantile(
        from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_tensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).quantile(
        from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_double_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile(
        from_raw::Tensor(self), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nanquantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_tensor_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_tensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nanquantile(
        from_raw::Tensor(self), from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_tensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).nanquantile(
        from_raw::Tensor(q), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_bool_intt_bool(void* values, void* indices, void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).sort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_bool_intt_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        from_raw::Tensor(self), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_bool_intt_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).sort(
        ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_bool_dimname_bool(void* values, void* indices, void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).sort(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_bool_dimname_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        from_raw::Tensor(self), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_bool_dimname_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).sort(
        ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_msort_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::msort_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_msort_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::msort(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_msort_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).msort(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argsort(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).argsort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::argsort(
        from_raw::Tensor(self), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).argsort(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool(void* values, void* indices, void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk_out(
        from_raw::Tensor(values), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_tensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_topk_tensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(from_raw::Tensor(self).topk(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::all(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).all(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::any(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).any(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_out_tensor_tensor_scalar_intt_scalar(void* out, void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::renorm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::renorm(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm_tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).renorm(
        ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unfold_tensor_intt_intt_intt(void* self, void* dimension, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).unfold(
        ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unfold_backward_tensor_intarrayref_intt_intt_intt(void* grad_in, void* input_sizes, void* dim, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::unfold_backward(
        from_raw::Tensor(grad_in), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::equal(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(from_raw::Tensor(self).equal(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_tensor_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow(
        from_raw::Tensor(self), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pow(
        from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_scalar_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_scalar_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow(
        ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_tensor_scalar(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pow(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pow(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pow_(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).pow_(
        from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_tensor_tensor_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power(
        from_raw::Tensor(self), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).float_power(
        from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_tensor_scalar_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_scalar_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power(
        ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_tensor_tensor_scalar(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::float_power(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).float_power(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power__tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).float_power_(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power__tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).float_power_(
        from_raw::Tensor(exponent)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_tensor_double_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::normal_out(
        from_raw::Tensor(out), from_raw::Tensor(mean), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_double_tensor_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::normal_out(
        from_raw::Tensor(out), ((LanternObject<double>*)mean)->get(), from_raw::Tensor(std), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_tensor_tensor_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::normal_out(
        from_raw::Tensor(out), from_raw::Tensor(mean), from_raw::Tensor(std), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_double_double_intarrayref_generator(void* out, void* mean, void* std, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::normal_out(
        from_raw::Tensor(out), ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alias_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::alias(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_alias_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).alias(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_index_copy_(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get(), from_raw::Tensor(index), from_raw::Tensor(source)));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cumsum(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cumsum_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cumprod(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cumprod_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__amp_foreach_non_finite_check_and_unscale__tensorlist_tensor_tensor(void* self, void* found_inf, void* inv_scale)
{
  LANTERN_FUNCTION_START
    torch::_amp_foreach_non_finite_check_and_unscale_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), from_raw::Tensor(found_inf), from_raw::Tensor(inv_scale));
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__amp_update_scale__tensor_tensor_tensor_double_double_intt(void* self, void* growth_tracker, void* found_inf, void* scale_growth_factor, void* scale_backoff_factor, void* growth_interval)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_amp_update_scale_(
        from_raw::Tensor(self), from_raw::Tensor(growth_tracker), from_raw::Tensor(found_inf), ((LanternObject<double>*)scale_growth_factor)->get(), ((LanternObject<double>*)scale_backoff_factor)->get(), ((LanternObject<int64_t>*)growth_interval)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_cat_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_tensorlist_scalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_add(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Scalar>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__tensorlist_scalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::Scalar>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_tensorlist_scalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sub(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Scalar>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__tensorlist_scalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::Scalar>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_tensorlist_scalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_mul(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Scalar>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__tensorlist_scalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::Scalar>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_tensorlist_scalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_div(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Scalar>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__tensorlist_scalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::Scalar>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_tensorlist_tensorlist_scalar(void* tensors1, void* tensors2, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_add(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__tensorlist_tensorlist_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_tensorlist_tensorlist_scalar(void* tensors1, void* tensors2, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sub(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__tensorlist_tensorlist_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_tensorlist_tensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_mul(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__tensorlist_tensorlist(void* self, void* other)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)other)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_tensorlist_tensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_div(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__tensorlist_tensorlist(void* self, void* other)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)other)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_tensorlist_arrayrefscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_add(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__tensorlist_arrayrefscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_tensorlist_arrayrefscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sub(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__tensorlist_arrayrefscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_tensorlist_arrayrefscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_div(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__tensorlist_arrayrefscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_tensorlist_arrayrefscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_mul(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__tensorlist_arrayrefscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_exp_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_exp(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_zero__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_zero_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_exp__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_exp_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sqrt_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sqrt(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sqrt__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sqrt_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_abs_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_abs(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_abs__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_abs_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_acos_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_acos(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_acos__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_acos_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_asin_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_asin(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_asin__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_asin_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_atan_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_atan(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_atan__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_atan_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_ceil_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_ceil(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_ceil__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_ceil_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cos_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_cos(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cos__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_cos_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cosh_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_cosh(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cosh__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_cosh_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erf_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_erf(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erf__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_erf_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erfc_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_erfc(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erfc__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_erfc_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_expm1_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_expm1(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_expm1__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_expm1_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_floor_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_floor(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_floor__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_floor_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_log(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log10_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_log10(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log10__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log10_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log1p_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_log1p(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log1p__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log1p_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log2_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_log2(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log2__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log2_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_neg_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_neg(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_neg__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_neg_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tan_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_tan(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tan__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_tan_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tanh_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_tanh(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tanh__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_tanh_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sin_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sin(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sin__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sin_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sinh_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sinh(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sinh__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sinh_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_round_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_round(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_round__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_round_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_lgamma_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_lgamma(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_lgamma__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_lgamma_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_frac_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_frac(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_frac__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_frac_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_reciprocal_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_reciprocal(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_reciprocal__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_reciprocal_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sigmoid_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sigmoid(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sigmoid__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sigmoid_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_trunc_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_trunc(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_trunc__tensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_trunc_(((LanternObject<std::vector<torch::Tensor>>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv__tensorlist_tensorlist_tensorlist_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcdiv_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul__tensorlist_tensorlist_tensorlist_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcmul_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv__tensorlist_tensorlist_tensorlist_arrayrefscalar(void* self, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcdiv_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul__tensorlist_tensorlist_tensorlist_arrayrefscalar(void* self, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcmul_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv_tensorlist_tensorlist_tensorlist_scalar(void* input, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_addcdiv(
        ((LanternObject<std::vector<torch::Tensor>>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul_tensorlist_tensorlist_tensorlist_scalar(void* input, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_addcmul(
        ((LanternObject<std::vector<torch::Tensor>>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv_tensorlist_tensorlist_tensorlist_arrayrefscalar(void* input, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_addcdiv(
        ((LanternObject<std::vector<torch::Tensor>>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul_tensorlist_tensorlist_tensorlist_arrayrefscalar(void* input, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_addcmul(
        ((LanternObject<std::vector<torch::Tensor>>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensor2)->get(), ((LanternObject<torch::ArrayRef<torch::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_maximum_tensorlist_tensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_maximum(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_minimum_tensorlist_tensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_minimum(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors1)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_tensor_tensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bucketize(
        from_raw::Tensor(self), from_raw::Tensor(boundaries), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_out_tensor_tensor_tensor_bool_bool(void* out, void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bucketize_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(boundaries), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_scalar_tensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::bucketize(
        ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(boundaries), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_tensor_tensor_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::searchsorted(
        from_raw::Tensor(sorted_sequence), from_raw::Tensor(self), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_out_tensor_tensor_tensor_bool_bool(void* out, void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::searchsorted_out(
        from_raw::Tensor(out), from_raw::Tensor(sorted_sequence), from_raw::Tensor(self), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_tensor_scalar_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::searchsorted(
        from_raw::Tensor(sorted_sequence), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mse_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mse_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mse_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mse_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::l1_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::l1_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::l1_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::l1_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* out, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multi_margin_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt(void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multi_margin_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multi_margin_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multi_margin_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multilabel_margin_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multilabel_margin_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt(void* output, void* is_target, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(is_target), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multilabel_margin_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), from_raw::Tensor(is_target)));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor(void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::multilabel_margin_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), from_raw::Tensor(is_target)));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_nd_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss_nd(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(total_weight), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), from_raw::Tensor(total_weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), from_raw::Tensor(total_weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss2d(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(total_weight), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), from_raw::Tensor(total_weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::nll_loss2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<c10::optional<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), from_raw::Tensor(total_weight)));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt_double(void* out, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::smooth_l1_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_tensor_tensor_intt_double(void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::smooth_l1_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::smooth_l1_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt_double(void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::smooth_l1_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_out_tensor_tensor_tensor_intt_double(void* out, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::huber_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_tensor_tensor_intt_double(void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::huber_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::huber_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_backward_tensor_tensor_tensor_intt_double(void* grad_output, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::huber_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::soft_margin_loss_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::soft_margin_loss(
        from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::soft_margin_loss_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::soft_margin_loss_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(target), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_out_tensor_tensor_scalar_scalar_scalar(void* out, void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::elu_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::elu(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_backward_tensor_scalar_scalar_scalar_bool_tensor(void* grad_output, void* alpha, void* scale, void* input_scale, void* is_result, void* self_or_result)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::elu_backward(
        from_raw::Tensor(grad_output), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get(), ((LanternObject<bool>*)is_result)->get(), from_raw::Tensor(self_or_result)));
  LANTERN_FUNCTION_END
}

void* _lantern_elu__tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::elu_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::glu_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::glu(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_out_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::glu_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_tensor_tensor_intt(void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::glu_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardsigmoid_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardsigmoid(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardsigmoid_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardsigmoid_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardtanh_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_tensor_scalar_scalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardtanh(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar(void* grad_input, void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardtanh_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_tensor_tensor_scalar_scalar(void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardtanh_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh__tensor_scalar_scalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardtanh_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardswish_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardswish(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardswish_(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::hardswish_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_out_tensor_tensor_scalar(void* out, void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::leaky_relu_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_tensor_scalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::leaky_relu(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_backward_tensor_tensor_scalar_bool(void* grad_output, void* self, void* negative_slope, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::leaky_relu_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)negative_slope)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu__tensor_scalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::leaky_relu_(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_sigmoid_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_sigmoid(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_out_tensor_tensor_tensor(void* output, void* buffer, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(buffer), from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_sigmoid_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(buffer)));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_tensor_tensor_tensor(void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::log_sigmoid_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(buffer)));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator(void* out, void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu_with_noise_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(noise), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu_with_noise(
        from_raw::Tensor(self), from_raw::Tensor(noise), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool_bool(void* grad_output, void* self, void* noise, void* lower, void* upper, void* training, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu_with_noise_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(noise), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::rrelu_with_noise_(
        from_raw::Tensor(self), from_raw::Tensor(noise), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_out_tensor_tensor_scalar_scalar(void* out, void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softplus_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_tensor_scalar_scalar(void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softplus(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor(void* grad_input, void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softplus_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor(void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softplus_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_out_tensor_tensor_scalar(void* out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softshrink_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softshrink(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_out_tensor_tensor_tensor_scalar(void* grad_input, void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softshrink_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_tensor_tensor_scalar(void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::softshrink_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_adaptive_avg_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_adaptive_avg_pool2d_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::mkldnn_adaptive_avg_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_adaptive_avg_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_adaptive_avg_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool3d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_adaptive_avg_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_avg_pool3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool3d_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_adaptive_avg_pool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d_out(
        from_raw::Tensor(out), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_max_pool2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_max_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d_out(
        from_raw::Tensor(out), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_max_pool3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::adaptive_max_pool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::avg_pool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d_out(
        from_raw::Tensor(output), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(random_samples))));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(random_samples))));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fractional_max_pool2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fractional_max_pool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d_out(
        from_raw::Tensor(output), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(random_samples))));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(random_samples))));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fractional_max_pool3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fractional_max_pool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices_out(
        from_raw::Tensor(out), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool2d_with_indices_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool2d_with_indices_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices_out(
        from_raw::Tensor(out), from_raw::Tensor(indices), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool3d_with_indices_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_pool3d_with_indices_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), from_raw::Tensor(indices)));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_tensor_tensor_intarrayref(void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool2d(
        from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref(void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref(void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool3d(
        from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::max_unpool3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(indices), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad1d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad1d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad1d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::reflection_pad2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad1d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad1d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad1d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::replication_pad3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d(
        from_raw::Tensor(input), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool_double(void* out, void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_tensor_intarrayref_bool_double(void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_linear1d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_tensor_intarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bilinear2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_tensor_intarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_bicubic2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool_double_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_tensor_intarrayref_bool_double_double_double(void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_double_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_trilinear3d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_out_tensor_tensor_intarrayref_double(void* out, void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_tensor_intarrayref_double(void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_double(void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest1d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_out_tensor_tensor_intarrayref_double_double(void* out, void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_tensor_intarrayref_double_double(void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_double_double(void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest2d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_out_tensor_tensor_intarrayref_double_double_double(void* out, void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_tensor_intarrayref_double_double_double(void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_double_double_double(void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::upsample_nearest3d_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sigmoid_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_tensor_tensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::sigmoid_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_out_tensor_tensor_tensor_double(void* grad_input, void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logit_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_tensor_tensor_double(void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::logit_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tanh_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_tensor_tensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::tanh_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(output)));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_transpose2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_transpose2d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_bias), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), from_raw::Tensor(columns), from_raw::Tensor(ones))));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), from_raw::Tensor(columns), from_raw::Tensor(ones), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_transpose3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_transpose3d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_bias), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input))));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv2d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_bias), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input))));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv_depthwise2d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv_depthwise2d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv_depthwise2d_forward_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::thnn_conv_depthwise2d_forward(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_weight, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::conv_depthwise3d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_depthwise3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_bias), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_depthwise3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv3d_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv3d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward_out(
        from_raw::Tensor(output), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_weight), from_raw::Tensor(grad_bias), from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input))));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), from_raw::Tensor(finput), from_raw::Tensor(fgrad_input), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_dilated2d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated2d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::slow_conv_dilated3d(
        from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated3d_backward(
        from_raw::Tensor(grad_output), from_raw::Tensor(self), from_raw::Tensor(weight), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::col2im_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::col2im(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::col2im_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::col2im_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_column_stack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::column_stack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_column_stack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::column_stack_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::im2col_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::im2col(
        from_raw::Tensor(self), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::im2col_backward_out(
        from_raw::Tensor(grad_input), from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::im2col_backward(
        from_raw::Tensor(grad_output), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isfinite_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isfinite(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isfinite_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isfinite(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isinf(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_record_stream_tensor_stream(void* self, void* s)
{
  LANTERN_FUNCTION_START
    from_raw::Tensor(self).record_stream(((LanternObject<at::Stream>*)s)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isposinf(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isposinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isposinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isposinf_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isneginf(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isneginf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).isneginf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::isneginf_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern__add_batch_dim_tensor_intt_intt(void* self, void* batch_dim, void* level)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_add_batch_dim(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)batch_dim)->get(), ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__remove_batch_dim_tensor_intt_intt_intt(void* self, void* level, void* batch_size, void* out_dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_remove_batch_dim(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)level)->get(), ((LanternObject<int64_t>*)batch_size)->get(), ((LanternObject<int64_t>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_entr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_entr(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_entr_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_entr_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expm1_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_expm1(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expm1_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_expm1_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_exp2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_exp2(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_exp2_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_exp2_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_gammaln_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_gammaln(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_gammaln_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_gammaln_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erf(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erf_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erfc(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erfc_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfinv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erfinv(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfinv_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_erfinv_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_scalar_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py(
        ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_tensor_scalar_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py_out(
        from_raw::Tensor(out), ((LanternObject<torch::Scalar>*)self)->get(), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_xlog1py_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_i0e_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_i0e(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_i0e_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_i0e_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_logit_tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_logit(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_logit_out_tensor_tensor_double(void* out, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_logit_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expit_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_expit(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expit_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::special_expit_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_hfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_hfft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_hfft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_hfft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ihfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ihfft(
        from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ihfft_out_tensor_tensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ihfft_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft2_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fft2(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft2_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fft2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft2_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifft2(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft2_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifft2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft2_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfft2(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft2_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfft2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft2_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfft2(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft2_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfft2_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fftn(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftn_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fftn_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifftn(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftn_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifftn_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfftn(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftn_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfftn_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfftn(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfftn_out_tensor_tensor_intarrayref_intarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_irfftn_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftfreq_intt_double_tensoroptions(void* n, void* d, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fftfreq(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftfreq_out_tensor_intt_double(void* out, void* n, void* d)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fftfreq_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftfreq_intt_double_tensoroptions(void* n, void* d, void* options)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfftfreq(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftfreq_out_tensor_intt_double(void* out, void* n, void* d)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_rfftfreq_out(
        from_raw::Tensor(out), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftshift_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_fftshift(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftshift_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::fft_ifftshift(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_ex_tensor_bool(void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_cholesky_ex(
        from_raw::Tensor(self), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_ex_out_tensor_tensor_tensor_bool(void* L, void* info, void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_cholesky_ex_out(
        from_raw::Tensor(L), from_raw::Tensor(info), from_raw::Tensor(self), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cholesky(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cholesky_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_det(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_det_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_det_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::det(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).det(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_lstsq_tensor_tensor_double_stdstring(void* self, void* b, void* rcond, void* driver)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_lstsq(
        from_raw::Tensor(self), from_raw::Tensor(b), ((LanternObject<c10::optional<double>>*)rcond)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(driver).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_lstsq_out_tensor_tensor_tensor_tensor_tensor_tensor_double_stdstring(void* solution, void* residuals, void* rank, void* singular_values, void* self, void* b, void* rcond, void* driver)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_lstsq_out(
        from_raw::Tensor(solution), from_raw::Tensor(residuals), from_raw::Tensor(rank), from_raw::Tensor(singular_values), from_raw::Tensor(self), from_raw::Tensor(b), ((LanternObject<c10::optional<double>>*)rcond)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(driver).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_slogdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_slogdet(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_slogdet_out_tensor_tensor_tensor(void* sign, void* logabsdet, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_slogdet_out(
        from_raw::Tensor(sign), from_raw::Tensor(logabsdet), from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eig_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eig(
        from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eig_out_tensor_tensor_tensor(void* eigenvalues, void* eigenvectors, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eig_out(
        from_raw::Tensor(eigenvalues), from_raw::Tensor(eigenvectors), from_raw::Tensor(self))));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvals_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_eigvals(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvals_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_eigvals_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigh_tensor_stdstring(void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eigh(
        from_raw::Tensor(self), ((LanternObject<std::string>*)UPLO)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigh_out_tensor_tensor_tensor_stdstring(void* eigvals, void* eigvecs, void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eigh_out(
        from_raw::Tensor(eigvals), from_raw::Tensor(eigvecs), from_raw::Tensor(self), ((LanternObject<std::string>*)UPLO)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvalsh_tensor_stdstring(void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_eigvalsh(
        from_raw::Tensor(self), ((LanternObject<std::string>*)UPLO)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvalsh_out_tensor_tensor_stdstring(void* out, void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_eigvalsh_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::string>*)UPLO)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_householder_product_tensor_tensor(void* input, void* tau)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_householder_product(
        from_raw::Tensor(input), from_raw::Tensor(tau)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_householder_product_out_tensor_tensor_tensor(void* out, void* input, void* tau)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_householder_product_out(
        from_raw::Tensor(out), from_raw::Tensor(input), from_raw::Tensor(tau)));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_inv_out_helper__tensor_tensor_tensor(void* self, void* infos_lu, void* infos_getri)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_linalg_inv_out_helper_(
        from_raw::Tensor(self), from_raw::Tensor(infos_lu), from_raw::Tensor(infos_getri)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_ex_tensor_bool(void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_inv_ex(
        from_raw::Tensor(self), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_ex_out_tensor_tensor_tensor_bool(void* inverse, void* info, void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_inv_ex_out(
        from_raw::Tensor(inverse), from_raw::Tensor(info), from_raw::Tensor(self), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_inv(
        from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_inv_out(
        from_raw::Tensor(out), from_raw::Tensor(self)));
  LANTERN_FUNCTION_END
}

void* _lantern_inner_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::inner(
        from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_inner_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).inner(
        from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_inner_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::inner_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::outer(
        from_raw::Tensor(self), from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_outer_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).outer(
        from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_out_tensor_tensor_tensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::outer_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ger(
        from_raw::Tensor(self), from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ger_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(from_raw::Tensor(self).ger(
        from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_out_tensor_tensor_tensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::ger_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(vec2)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_norm(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(ord).get())->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_tensor_stdstring_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_norm(
        from_raw::Tensor(self), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(ord).get())->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_tensor_tensor_stdstring_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_vector_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_vector_norm(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_vector_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_vector_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_norm(
        from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)ord)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<torch::Scalar>*)ord)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_tensor_stdstring_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_norm(
        from_raw::Tensor(self), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_out_tensor_tensor_stdstring_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_norm_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svd_out_tensor_tensor_tensor_tensor_bool(void* U, void* S, void* Vh, void* self, void* full_matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_svd_out(
        from_raw::Tensor(U), from_raw::Tensor(S), from_raw::Tensor(Vh), from_raw::Tensor(self), ((LanternObject<bool>*)full_matrices)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svd_tensor_bool(void* self, void* full_matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_svd(
        from_raw::Tensor(self), ((LanternObject<bool>*)full_matrices)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svdvals_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_svdvals(
        from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svdvals_out_tensor_tensor(void* out, void* input)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_svdvals_out(
        from_raw::Tensor(out), from_raw::Tensor(input)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cond(
        from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_out_tensor_tensor_scalar(void* out, void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cond_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_tensor_stdstring(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cond(
        from_raw::Tensor(self), ((LanternObject<std::string>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_out_tensor_tensor_stdstring(void* out, void* self, void* p)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_cond_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<std::string>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_tensor_double_bool(void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_pinv(
        from_raw::Tensor(self), ((LanternObject<double>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_tensor_tensor_bool(void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_pinv(
        from_raw::Tensor(self), from_raw::Tensor(rcond), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_out_tensor_tensor_double_bool(void* out, void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_pinv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<double>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_out_tensor_tensor_tensor_bool(void* out, void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_pinv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(rcond), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_solve_out_helper__tensor_tensor_tensor(void* self, void* other, void* infos)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_linalg_solve_out_helper_(
        from_raw::Tensor(self), from_raw::Tensor(other), from_raw::Tensor(infos)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_solve_tensor_tensor(void* input, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_solve(
        from_raw::Tensor(input), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_solve_out_tensor_tensor_tensor(void* out, void* input, void* other)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_solve_out(
        from_raw::Tensor(out), from_raw::Tensor(input), from_raw::Tensor(other)));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorinv_tensor_intt(void* self, void* ind)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_tensorinv(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)ind)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorinv_out_tensor_tensor_intt(void* out, void* self, void* ind)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_tensorinv_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)ind)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorsolve_tensor_tensor_intarrayref(void* self, void* other, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_tensorsolve(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorsolve_out_tensor_tensor_tensor_intarrayref(void* out, void* self, void* other, void* dims)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_tensorsolve_out(
        from_raw::Tensor(out), from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_qr_tensor_stdstring(void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_qr(
        from_raw::Tensor(self), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_qr_out_tensor_tensor_tensor_stdstring(void* Q, void* R, void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_qr_out(
        from_raw::Tensor(Q), from_raw::Tensor(R), from_raw::Tensor(self), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_qr_helper_tensor_stdstring(void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_linalg_qr_helper(
        from_raw::Tensor(self), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_power_tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_power(
        from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_power_out_tensor_tensor_intt(void* out, void* self, void* n)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_power_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_tensor_double_bool(void* self, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_rank(
        from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_out_tensor_tensor_double_bool(void* out, void* self, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_rank_out(
        from_raw::Tensor(out), from_raw::Tensor(self), ((LanternObject<c10::optional<double>>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_tensor_tensor_bool(void* input, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_rank(
        from_raw::Tensor(input), from_raw::Tensor(tol), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_out_tensor_tensor_tensor_bool(void* out, void* input, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_matrix_rank_out(
        from_raw::Tensor(out), from_raw::Tensor(input), from_raw::Tensor(tol), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_multi_dot_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_multi_dot(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_multi_dot_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::linalg_multi_dot_out(
        from_raw::Tensor(out), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_serialization_subcmul_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_serialization_subcmul(
        from_raw::Tensor(self), from_raw::Tensor(other), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_intlist_tensor_intarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_optional_intlist(
        from_raw::Tensor(values), ((LanternObject<c10::optional<torch::IntArrayRef>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_filled_intlist_tensor_intarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_optional_filled_intlist(
        from_raw::Tensor(values), ((LanternObject<c10::optional<torch::IntArrayRef>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_floatlist_tensor_arrayrefdouble(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_optional_floatlist(
        from_raw::Tensor(values), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_string_default_tensor_stdstring_stdstring(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_string_default(
        from_raw::Tensor(dummy), ((LanternObject<std::string>*)a)->get(), ((LanternObject<std::string>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_ambiguous_defaults_tensor_intt_intt(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_ambiguous_defaults(
        from_raw::Tensor(dummy), ((LanternObject<int64_t>*)a)->get(), ((LanternObject<int64_t>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_ambiguous_defaults_tensor_intt_stdstring(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::_test_ambiguous_defaults(
        from_raw::Tensor(dummy), ((LanternObject<int64_t>*)a)->get(), ((LanternObject<std::string>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_segment_reduce_tensor_stdstring_tensor_tensor_intt_bool_scalar(void* data, void* reduce, void* lengths, void* indices, void* axis, void* unsafe, void* initial)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::segment_reduce(
        from_raw::Tensor(data), ((LanternObject<std::string>*)reduce)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)lengths)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)indices)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<bool>*)unsafe)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(initial).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_segment_reduce_backward_tensor_tensor_tensor_tensor(void* grad, void* output, void* data, void* lengths)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::segment_reduce_backward(
        from_raw::Tensor(grad), from_raw::Tensor(output), from_raw::Tensor(data), ((LanternObject<c10::optional<torch::Tensor>>*)lengths)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pad_sequence_tensorlist_bool_double(void* sequences, void* batch_first, void* padding_value)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::pad_sequence(
        ((LanternObject<std::vector<torch::Tensor>>*)sequences)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)padding_value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_dense_tensors_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return make_unique::Tensor(torch::flatten_dense_tensors(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unflatten_dense_tensors_tensor_tensorlist(void* flat, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unflatten_dense_tensors(
        from_raw::Tensor(flat), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

/* Autogen Body -- End */
