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
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Byte(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_char_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Char(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_double_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Double(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_float_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Float(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_int_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Int(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_long_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Long(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_short_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Short(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_half_tensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cast_Half(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_backward_tensor_tensor_bool_bool(void* self, void* gradient, void* retain_graph, void* create_graph)
{
  LANTERN_FUNCTION_START
    ((LanternObject<torch::Tensor>*)self)->get().backward(((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(gradient).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(retain_graph).get())->get(), ((LanternObject<bool>*)create_graph)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set_data_tensor_tensor(void* self, void* new_data)
{
  LANTERN_FUNCTION_START
    ((LanternObject<torch::Tensor>*)self)->get().set_data(((LanternObject<torch::Tensor>*)new_data)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_data_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().data(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_leaf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_leaf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_output_nr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().output_nr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__version_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get()._version(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_requires_grad__tensor_bool(void* self, void* requires_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().requires_grad_(
        ((LanternObject<bool>*)requires_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_retain_grad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    ((LanternObject<torch::Tensor>*)self)->get().retain_grad();
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename__tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rename_(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rename(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().align_to(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_tensor_dimnamelist_intt(void* self, void* order, void* ellipsis_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().align_to(
        ((LanternPtr<std::vector<torch::Dimname>>*)order)->get(), ((LanternObject<int64_t>*)ellipsis_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().align_as(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_align_tensors_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::align_tensors(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_refine_names_tensor_dimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().refine_names(
        ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_ctc_loss(
        ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* deterministic, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_ctc_loss(
        ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_rnn_flatten_weight()
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_rnn_flatten_weight(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_flatten_weight_tensorlist_intt_intt_intt_intt_intt_bool_bool(void* weight_arr, void* weight_stride0, void* input_size, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cudnn_rnn_flatten_weight(
        ((LanternObject<std::vector<torch::Tensor>>*)weight_arr)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<int64_t>*)input_size)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<bool>*)bidirectional)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_tensor_tensorlist_intt_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight_buf).get())->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(cx).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(dropout_state).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn_backward(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<torch::Tensor>*)weight_buf)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(cx).get())->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_output).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_cy).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(dropout_state).get())->get(), ((LanternObject<torch::Tensor>*)reserve)->get(), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_init_dropout_state_double_bool_intt_tensoroptions(void* dropout, void* train, void* dropout_seed, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cudnn_init_dropout_state(
        ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<int64_t>*)dropout_seed)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__debug_has_internal_overlap_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::_debug_has_internal_overlap(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fused_dropout_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fused_dropout(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__masked_scale_tensor_tensor_double(void* self, void* mask, void* scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_masked_scale(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<double>*)scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_draw_tensor_intt_tensor_intt_intt_scalartype(void* quasi, void* n, void* sobolstate, void* dimension, void* num_generated, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_sobol_engine_draw(
        ((LanternObject<torch::Tensor>*)quasi)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::Tensor>*)sobolstate)->get(), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_ff__tensor_intt_tensor_intt_intt(void* self, void* n, void* sobolstate, void* dimension, void* num_generated)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_ff_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::Tensor>*)sobolstate)->get(), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_scramble__tensor_tensor_intt(void* self, void* ltm, void* dimension)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_scramble_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)ltm)->get(), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_initialize_state__tensor_intt(void* self, void* dimension)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sobol_engine_initialize_state_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__reshape_from_tensor_tensor_tensor(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_reshape_from_tensor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__shape_as_tensor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_shape_as_tensor(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dropout(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dropout_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::feature_dropout(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::feature_dropout_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::alpha_dropout(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::alpha_dropout_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout_tensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::feature_alpha_dropout(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout__tensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::feature_alpha_dropout_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::abs(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().abs(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::abs_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().abs_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::abs_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::absolute(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().absolute(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().absolute_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::absolute_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::angle(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_angle_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().angle(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::angle_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_real_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::view_as_real(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::view_as_complex(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sgn(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sgn(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sgn_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sgn_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_real_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::real(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_imag_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::imag(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conj(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().conj(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conj_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__conj_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_conj(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acos(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().acos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acos_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().acos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acos_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccos(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arccos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccos_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arccos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccos_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool1d_tensor_intarrayref_intarrayref_intarrayref_bool_bool(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool1d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool1d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_add_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().add(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().add_(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::add_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_add_relu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_add_relu_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_add_relu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().add(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().add_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addmv(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv_tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addmv(
        ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv__tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addmv_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv__tensor_tensor_tensor_scalar_scalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addmv_(
        ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addmv_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__addmv_impl__tensor_tensor_tensor_tensor_scalar_scalar(void* self, void* self2, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_addmv_impl_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)self2)->get(), ((LanternObject<torch::Tensor>*)mat)->get(), ((LanternObject<torch::Tensor>*)vec)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addr(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec1)->get(), ((LanternObject<torch::Tensor>*)vec2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr_tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addr(
        ((LanternObject<torch::Tensor>*)vec1)->get(), ((LanternObject<torch::Tensor>*)vec2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr__tensor_tensor_tensor_scalar_scalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addr_(
        ((LanternObject<torch::Tensor>*)vec1)->get(), ((LanternObject<torch::Tensor>*)vec2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addr_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec1)->get(), ((LanternObject<torch::Tensor>*)vec2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_tensor_intarrayref_bool(void* theta, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::affine_grid_generator(
        ((LanternObject<torch::Tensor>*)theta)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_backward_tensor_intarrayref_bool(void* grad, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::affine_grid_generator_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::all(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().all(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::all_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::all(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().all(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::all_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_allclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::allclose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_allclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().allclose(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::any(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().any(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_tensor_tensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::any_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::any(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().any(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_tensor_tensor_dimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::any_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_tensoroptions(void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arange(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_tensor_scalar(void* out, void* end)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arange_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)end)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arange_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dim_arange_tensor_intt(void* like, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_dim_arange(
        ((LanternObject<torch::Tensor>*)like)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::argmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().argmax(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmin_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::argmin(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmin_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().argmin(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acosh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().acosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acosh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().acosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::acosh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccosh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arccosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccosh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arccosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arccosh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asinh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().asinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asinh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().asinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asinh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsinh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arcsinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsinh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arcsinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsinh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atanh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atanh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atanh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctanh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arctanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctanh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arctanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctanh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided_tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::as_strided(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided_tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().as_strided(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided__tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::as_strided_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided__tensor_intarrayref_intarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().as_strided_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asin(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().asin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asin_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().asin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::asin_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsin(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arcsin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsin_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arcsin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arcsin_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atan(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atan_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atan_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctan(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arctan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctan_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().arctan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::arctan_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_1d_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atleast_1d(
        ((LanternObject<torch::Tensor>*)self)->get()));
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
    return (void *) new LanternObject<torch::Tensor>(torch::atleast_2d(
        ((LanternObject<torch::Tensor>*)self)->get()));
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
    return (void *) new LanternObject<torch::Tensor>(torch::atleast_3d(
        ((LanternObject<torch::Tensor>*)self)->get()));
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
    return (void *) new LanternObject<torch::Tensor>(torch::baddbmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().baddbmm(
        ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().baddbmm_(
        ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__baddbmm_mkl__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_baddbmm_mkl_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_baddbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::baddbmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::batch_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_batch_norm_tensor_tensor_tensor_tensor_tensor_double_double_intt(void* input, void* weight, void* bias, void* mean, void* var, void* eps, void* output_scale, void* output_zero_point)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_batch_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)var)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<double>*)output_scale)->get(), ((LanternObject<int64_t>*)output_zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_backward_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool_tensor(void* impl_index, void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var_transform, void* train, void* eps, void* output_mask, void* reservedSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index_backward(
        ((LanternObject<int64_t>*)impl_index)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_var_transform).get())->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get(), ((LanternObject<torch::Tensor>*)reservedSpace)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bernoulli(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bernoulli(
        ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_out_tensor_tensor_generator(void* out, void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bernoulli_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__tensor_tensor_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bernoulli_(
        ((LanternObject<torch::Tensor>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bernoulli_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bernoulli(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bernoulli(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bilinear_tensor_tensor_tensor_tensor(void* input1, void* input2, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bilinear(
        ((LanternObject<torch::Tensor>*)input1)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_out_tensor_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_out_tensor_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_tensor_tensor_tensor_tensor_intt(void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_with_logits(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(pos_weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binary_cross_entropy_with_logits_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(pos_weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bincount_tensor_tensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bincount(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weights).get())->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bincount_tensor_tensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bincount(
        ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weights).get())->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_not(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_not_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_not(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_not_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_xor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_xor(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_xor_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_xor_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_and(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_and(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_and_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_and_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_or(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_or(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logical_or_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logical_or_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bmm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bmm(
        ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_tensor_tensor_bool(void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_bmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_out_tensor_tensor_tensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_out_tensor_tensor_tensor_bool(void* out, void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_bmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_broadcast_tensors_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::broadcast_tensors(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cat_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_tensorlist_dimname(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_tensor_tensorlist_dimname(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cat_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_block_diag_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::block_diag(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ceil(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ceil(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ceil_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ceil_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ceil_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chain_matmul_tensorlist(void* matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::chain_matmul(
        ((LanternObject<std::vector<torch::Tensor>>*)matrices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_chunk(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().unsafe_chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::chunk(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_chunk_tensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp_(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_max(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max_tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp_max(
        ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max__tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_max_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max__tensor_scalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp_max_(
        ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_out_tensor_tensor_scalar(void* out, void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_max_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_min(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min_tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp_min(
        ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min__tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_min_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min__tensor_scalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clamp_min_(
        ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_out_tensor_tensor_scalar(void* out, void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clamp_min_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clip(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip_tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clip(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clip_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip__tensor_scalar_scalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clip_(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clip_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(min).get())->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_is_acceptable_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::cudnn_is_acceptable(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_tensor_tensor(void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::complex(
        ((LanternObject<torch::Tensor>*)real)->get(), ((LanternObject<torch::Tensor>*)imag)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_out_tensor_tensor_tensor(void* out, void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::complex_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)real)->get(), ((LanternObject<torch::Tensor>*)imag)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_tensor_tensor(void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::polar(
        ((LanternObject<torch::Tensor>*)abs)->get(), ((LanternObject<torch::Tensor>*)angle)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_out_tensor_tensor_tensor(void* out, void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::polar_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)abs)->get(), ((LanternObject<torch::Tensor>*)angle)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_constant_pad_nd_tensor_intarrayref_scalar(void* self, void* pad, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::constant_pad_nd(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)pad)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_contiguous_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().contiguous(
        ((LanternObject<torch::MemoryFormat>*)memory_format)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::convolution(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::convolution_overrideable(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_backward_overrideable_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_stdarraybool(void* grad_output, void* input, void* weight, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::convolution_backward_overrideable(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_convolution(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_convolution(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_nogroup_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_convolution_nogroup(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_double_backward_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_intarrayref_intt_bool_bool_bool_bool_stdarraybool(void* ggI, void* ggW, void* ggb, void* gO, void* weight, void* self, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_convolution_double_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(ggI).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(ggW).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(ggb).get())->get(), ((LanternObject<torch::Tensor>*)gO)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv1d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv3d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_tensor_tensor_tensor_intt(void* self, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv_tbc(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<int64_t>*)pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_backward_tensor_tensor_tensor_tensor_intt(void* self, void* input, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_tbc_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<int64_t>*)pad)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose1d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose1d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose2d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose3d_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_intarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::conv_transpose3d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copy__tensor_tensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().copy_(
        ((LanternObject<torch::Tensor>*)src)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__copy_from_tensor_tensor_bool(void* self, void* dst, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_copy_from(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)dst)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cos(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cos_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cos_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cosh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cosh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cosh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_embedding_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cosine_embedding_loss(
        ((LanternObject<torch::Tensor>*)input1)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::count_nonzero(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().count_nonzero(
        ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::count_nonzero(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().count_nonzero(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_tensor_intt_intt_intt_intt(void* theta, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_affine_grid_generator(
        ((LanternObject<torch::Tensor>*)theta)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_backward_tensor_intt_intt_intt_intt(void* grad, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_affine_grid_generator_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double_tensor(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon, void* reserveSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm_backward(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_var).get())->get(), ((LanternObject<double>*)epsilon)->get(), ((LanternObject<torch::Tensor>*)reserveSpace)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_transpose_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_transpose_backward_input(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_convolution_transpose_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_tensor_tensor(void* self, void* grid)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cudnn_grid_sampler(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grid)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_backward_tensor_tensor_tensor(void* self, void* grid, void* grad_output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_grid_sampler_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().cummax(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_tensor_tensor_tensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().cummax(
        ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_tensor_tensor_tensor_dimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummax_helper_tensor_tensor_tensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummax_helper(((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().cummin(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_tensor_tensor_tensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().cummin(
        ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_tensor_tensor_tensor_dimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummin_helper_tensor_tensor_tensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummin_helper(((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummaxmin_backward_tensor_tensor_tensor_intt(void* grad, void* input, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cummaxmin_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cumprod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cumprod(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_backward_tensor_tensor_intt(void* grad, void* input, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumprod_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cumsum(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_tensor_tensor_intt_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cumsum(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_tensor_tensor_dimname_scalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cumsum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ctc_loss(
        ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_tensor_tensor_tensor_tensor_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ctc_loss(
        ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<torch::Tensor>*)input_lengths)->get(), ((LanternObject<torch::Tensor>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_tensor_tensor_intarrayref_intarrayref_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_ctc_loss(
        ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_backward_tensor_tensor_tensor_intarrayref_intarrayref_tensor_tensor_intt_bool(void* grad, void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* neg_log_likelihood, void* log_alpha, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_ctc_loss_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)log_probs)->get(), ((LanternObject<torch::Tensor>*)targets)->get(), ((LanternObject<std::vector<int64_t>>*)input_lengths)->get(), ((LanternObject<std::vector<int64_t>>*)target_lengths)->get(), ((LanternObject<torch::Tensor>*)neg_log_likelihood)->get(), ((LanternObject<torch::Tensor>*)log_alpha)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_embed_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diag_embed(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_embed_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().diag_embed(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagflat_tensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diagflat(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagflat_tensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().diagflat(
        ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diagonal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_tensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().diagonal(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_tensor_dimname_dimname_dimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diagonal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)outdim)->get(), ((LanternObject<torch::Dimname>*)dim1)->get(), ((LanternObject<torch::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_tensor_dimname_dimname_dimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().diagonal(
        ((LanternObject<torch::Dimname>*)outdim)->get(), ((LanternObject<torch::Dimname>*)dim1)->get(), ((LanternObject<torch::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_backward_tensor_intarrayref_intt_intt_intt(void* grad, void* input_sizes, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diagonal_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill_diagonal__tensor_scalar_bool(void* self, void* fill_value, void* wrap)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fill_diagonal_(
        ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<bool>*)wrap)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::div(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().div(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().div_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::div_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::div(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().div(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().div_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().divide(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().divide_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::divide_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().divide(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::true_divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().true_divide(
        ((LanternObject<torch::Tensor>*)other)->get()));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().true_divide_(
        ((LanternObject<torch::Tensor>*)other)->get()));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::true_divide_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::true_divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
#ifdef CUDA102
    throw "Not Implemented";
#else
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().true_divide(
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
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().true_divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
#endif
  LANTERN_FUNCTION_END
}

void* _lantern_dot_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dot(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dot_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().dot(
        ((LanternObject<torch::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dot_out_tensor_tensor_tensor(void* out, void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dot_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::vdot(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vdot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().vdot(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::vdot_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_einsum_stdstring_tensorlist(void* equation, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::einsum(
        ((LanternObject<std::string>*)equation)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_tensor_tensor_intt_bool_bool(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::embedding(
        ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_backward_tensor_tensor_intt_intt_bool_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_dense_backward_tensor_tensor_intt_intt_bool(void* grad_output, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_dense_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_renorm__tensor_tensor_double_double(void* self, void* indices, void* max_norm, void* norm_type)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_renorm_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<double>*)max_norm)->get(), ((LanternObject<double>*)norm_type)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_sparse_backward_tensor_tensor_intt_intt_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::embedding_sparse_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_forward_only_tensor_tensor_tensor_bool_intt_bool_tensor_bool(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag_forward_only(
        ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::embedding_bag(
        ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_tensor_tensor_tensor_bool_intt_bool_tensor_bool(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag(
        ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_bool_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<torch::Tensor>*)offset2bag)->get(), ((LanternObject<torch::Tensor>*)bag_size)->get(), ((LanternObject<torch::Tensor>*)maximum_indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_sparse_backward_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_sparse_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<torch::Tensor>*)offset2bag)->get(), ((LanternObject<torch::Tensor>*)bag_size)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_dense_backward_tensor_tensor_tensor_tensor_tensor_tensor_intt_bool_intt_tensor(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_dense_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<torch::Tensor>*)offset2bag)->get(), ((LanternObject<torch::Tensor>*)bag_size)->get(), ((LanternObject<torch::Tensor>*)maximum_indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(per_sample_weights).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_per_sample_weights_backward_tensor_tensor_tensor_tensor_tensor_intt(void* grad, void* weight, void* indices, void* offsets, void* offset2bag, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_embedding_bag_per_sample_weights_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)offsets)->get(), ((LanternObject<torch::Tensor>*)offset2bag)->get(), ((LanternObject<int64_t>*)mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_meta_intarrayref_tensoroptions_memoryformat(void* size, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty_meta(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_intarrayref_dimnamelist_tensoroptions_memoryformat(void* size, void* names, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_intarrayref_tensoroptions_memoryformat(void* size, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_empty_tensor_intarrayref_tensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().new_empty(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_full_tensor_intarrayref_scalar_tensoroptions(void* self, void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().new_full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_zeros_tensor_intarrayref_tensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().new_zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_affine_quantized_intarrayref_tensoroptions_double_intt_memoryformat(void* size, void* options, void* scale, void* zero_point, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_empty_affine_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_per_channel_affine_quantized_intarrayref_tensor_tensor_intt_tensoroptions_memoryformat(void* size, void* scales, void* zero_points, void* axis, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_empty_per_channel_affine_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Tensor>*)scales)->get(), ((LanternObject<torch::Tensor>*)zero_points)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize__tensor_intarrayref_memoryformat(void* self, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().resize_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_quantized_intarrayref_tensor(void* size, void* qtensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty_quantized(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Tensor>*)qtensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_out_tensor_intarrayref_memoryformat(void* out, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_strided_intarrayref_intarrayref_tensoroptions(void* size, void* stride, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::empty_strided(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erf(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erf_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erf_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erf_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erfc(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erfc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erfc_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erfc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erfc_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().exp_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp2(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().exp2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp2_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().exp2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::exp2_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::expm1(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().expm1(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::expm1_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().expm1_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::expm1_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_tensor_intarrayref_bool(void* self, void* size, void* implicit)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().expand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<bool>*)implicit)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().expand_as(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_tensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_intt_tensoroptions(void* n, void* m, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_tensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eye_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_tensor_intt_intt(void* out, void* n, void* m)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eye_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flatten(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_intt_intt_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flatten(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_intt_intt_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_dimname_dimname_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flatten(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)start_dim)->get(), ((LanternObject<torch::Dimname>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_dimname_dimname_dimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flatten(
        ((LanternObject<torch::Dimname>*)start_dim)->get(), ((LanternObject<torch::Dimname>*)end_dim)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_tensor_dimnamelist_dimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flatten(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dims)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_tensor_dimnamelist_dimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flatten(
        ((LanternPtr<std::vector<torch::Dimname>>*)dims)->get(), ((LanternObject<torch::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_tensor_intt_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().unflatten(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_tensor_dimname_intarrayref_dimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().unflatten(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__tensor_scalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fill_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__tensor_scalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fill_(
        ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__tensor_tensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fill_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__tensor_tensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fill_(
        ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor_divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor_divide(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor_divide_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor_divide_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::floor_divide(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor_divide(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().floor_divide_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frac(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().frac(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frac_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().frac_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frac_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_intarrayref_scalar_dimnamelist_tensoroptions(void* size, void* fill_value, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_intarrayref_scalar_tensoroptions(void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::full(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_out_tensor_intarrayref_scalar(void* out, void* size, void* fill_value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::full_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_like_tensor_scalar_tensoroptions_memoryformat(void* self, void* fill_value, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::full_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)fill_value)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_from_file_stdstring_bool_intt_tensoroptions(void* filename, void* shared, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::from_file(
        ((LanternObject<std::string>*)filename)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(shared).get())->get(), ((LanternObject<c10::optional<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gcd_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gcd(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gcd(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gcd_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gcd_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lcm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lcm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lcm(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lcm_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lcm_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::grid_sampler(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::grid_sampler_2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_grid_sampler_2d_cpu_fallback(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_grid_sampler_2d_cpu_fallback_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_tensor_tensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::grid_sampler_3d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_backward_tensor_tensor_tensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_double_tensoroptions(void* window_length, void* periodic, void* alpha, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_tensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_tensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_double_tensoroptions(void* window_length, void* periodic, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hinge_embedding_loss_tensor_tensor_double_intt(void* self, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hinge_embedding_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_group_norm_tensor_intt_tensor_tensor_double_bool(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::group_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<int64_t>*)num_groups)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_tensor_tensor_tensor_intt_intt_intt_intt_double(void* input, void* weight, void* bias, void* N, void* C, void* HxW, void* group, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_intt_stdarraybool(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* N, void* C, void* HxW, void* group, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm_backward(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)rstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_ifft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ifft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ifft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ifft(
        ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rfft_tensor_intt_bool_bool(void* self, void* signal_ndim, void* normalized, void* onesided)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rfft_tensor_intt_bool_bool(void* self, void* signal_ndim, void* normalized, void* onesided)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rfft(
        ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_irfft_tensor_intt_bool_bool_intarrayref(void* self, void* signal_ndim, void* normalized, void* onesided, void* signal_sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::irfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<bool>*)onesided)->get(), ((LanternObject<std::vector<int64_t>>*)signal_sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_irfft_tensor_intt_bool_bool_intarrayref(void* self, void* signal_ndim, void* normalized, void* onesided, void* signal_sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().irfft(
        ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<bool>*)onesided)->get(), ((LanternObject<std::vector<int64_t>>*)signal_sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_bool_bool_intarrayref(void* self, void* signal_ndim, void* complex_input, void* complex_output, void* inverse, void* checked_signal_sizes, void* normalized, void* onesided, void* output_sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_fft_with_size(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)complex_input)->get(), ((LanternObject<bool>*)complex_output)->get(), ((LanternObject<bool>*)inverse)->get(), ((LanternObject<std::vector<int64_t>>*)checked_signal_sizes)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<bool>*)onesided)->get(), ((LanternObject<std::vector<int64_t>>*)output_sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_with_size_tensor_intt_bool_bool_bool_intarrayref_intt_bool_intarrayref(void* self, void* signal_ndim, void* complex_input, void* complex_output, void* inverse, void* checked_signal_sizes, void* normalization, void* onesided, void* output_sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_fft_with_size(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)complex_input)->get(), ((LanternObject<bool>*)complex_output)->get(), ((LanternObject<bool>*)inverse)->get(), ((LanternObject<std::vector<int64_t>>*)checked_signal_sizes)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)onesided)->get(), ((LanternObject<std::vector<int64_t>>*)output_sizes)->get()));
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

void* _lantern_index_tensor_tensorlist(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_tensor_tensorlist(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index(
        ((LanternObject<std::vector<torch::Tensor>>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_copy_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_copy(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_copy(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_copy_(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_copy(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_copy(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put__tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_put_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put__tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_put_(
        ((LanternObject<std::vector<torch::Tensor>>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put_tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_put(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put_tensor_tensorlist_tensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_put(
        ((LanternObject<std::vector<torch::Tensor>>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__index_put_impl__tensor_tensorlist_tensor_bool_bool(void* self, void* indices, void* values, void* accumulate, void* unsafe)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_index_put_impl_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<torch::Tensor>>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get(), ((LanternObject<bool>*)unsafe)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_instance_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::instance_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)use_input_stats)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::inverse(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_inverse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().inverse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::inverse_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__inverse_helper_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_inverse_helper(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isclose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isclose_tensor_tensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isclose(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isnan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isnan(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isnan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isnan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_distributed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_distributed(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_distributed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_distributed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_floating_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_floating_point(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_floating_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_floating_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_complex(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_complex_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_complex(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isreal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isreal(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isreal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isreal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_nonzero(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_same_size_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_same_size(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_same_size_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_same_size(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_is_signed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_signed(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_signed_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_signed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_tensor_tensor_intt_bool(void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::kl_div(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_backward_tensor_tensor_tensor_intt_bool(void* grad_output, void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::kl_div_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_tensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_tensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_tensor_tensor_tensor_intt_intt_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_tensor_intt_dimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_tensor_intt_dimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_tensor_tensor_tensor_intt_dimname_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_layer_norm_tensor_intarrayref_tensor_tensor_double_bool(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::layer_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<int64_t>>*)normalized_shape)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enable)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_tensor_tensor_tensor_intt_intt_double(void* input, void* weight, void* bias, void* M, void* N, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<int64_t>*)M)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_backward_tensor_tensor_tensor_tensor_tensor_intt_intt_stdarraybool(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* M, void* N, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm_backward(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)rstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)M)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linear_tensor_tensor_tensor(void* input, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linear(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_tensor_tensor_tensor(void* input, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_linear(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_fp32_activation_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_int8_weight_fp32_activation(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)packed)->get(), ((LanternObject<torch::Tensor>*)col_offsets)->get(), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), ((LanternObject<torch::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_tensor_tensor_tensor_tensor_scalar_scalar_tensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_int8_weight(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)packed)->get(), ((LanternObject<torch::Tensor>*)col_offsets)->get(), ((LanternObject<torch::Scalar>*)weight_scale)->get(), ((LanternObject<torch::Scalar>*)weight_zero_point)->get(), ((LanternObject<torch::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_quantize_weight_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fbgemm_linear_quantize_weight(
        ((LanternObject<torch::Tensor>*)input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_gemm_matrix_fp16_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_gemm_matrix_fp16(
        ((LanternObject<torch::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_fp32_activation_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_fp16_weight_fp32_activation(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)packed_weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_tensor_tensor_tensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_linear_fp16_weight(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)packed_weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_tensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<torch::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_tensor_intt_intt(void* input, void* K, void* N)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<int64_t>*)K)->get(), ((LanternObject<int64_t>*)N)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_scalar_scalar_intt_tensoroptions(void* start, void* end, void* steps, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_out_tensor_scalar_scalar_intt(void* out, void* start, void* end, void* steps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linspace_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log10(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log10(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log10_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log10_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log10_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log1p(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log1p(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log1p_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log1p_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log1p_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log2(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log2_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log2_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logaddexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logaddexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logaddexp(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logaddexp2_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logaddexp2(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logaddexp2(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logdet(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logdet(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_scalar_scalar_intt_double_tensoroptions(void* start, void* end, void* steps, void* base, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logspace(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_out_tensor_scalar_scalar_intt_double(void* out, void* start, void* end, void* steps, void* base)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logspace_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log_softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log_softmax(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_log_softmax_backward_data(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_logcumsumexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_logcumsumexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logcumsumexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logcumsumexp(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logcumsumexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logcumsumexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logcumsumexp(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_tensor_tensor_dimname(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logcumsumexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logsumexp(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_tensor_dimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logsumexp(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_tensor_tensor_dimnamelist_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logsumexp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_margin_ranking_loss_tensor_tensor_tensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::margin_ranking_loss(
        ((LanternObject<torch::Tensor>*)input1)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matmul(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matmul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().matmul(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matmul_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_tensor_double_bool(void* self, void* tol, void* symmetric)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_rank(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)tol)->get(), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_tensor_bool(void* self, void* symmetric)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_rank(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_power_tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_power(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_power_tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().matrix_power(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_exp(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_exp_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().matrix_exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_backward_tensor_tensor(void* self, void* grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::matrix_exp_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_tensor_tensor(void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_compute_linear_combination(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)coefficients)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_out_tensor_tensor_tensor(void* out, void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_compute_linear_combination_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)coefficients)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().max(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor_intt_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        ((LanternObject<torch::Tensor>*)max)->get(), ((LanternObject<torch::Tensor>*)max_values)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().max(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor_dimname_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        ((LanternObject<torch::Tensor>*)max)->get(), ((LanternObject<torch::Tensor>*)max_values)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_value_selecting_reduction_backward_tensor_intt_tensor_intarrayref_bool(void* grad, void* dim, void* indices, void* sizes, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::value_selecting_reduction_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::amax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amax_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().amax(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::amax_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool1d_with_indices(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_max_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_max_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool1d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_max_pool1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool2d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_max_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mean(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mean(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mean_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mean(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mean_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().median(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().median(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().min(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor_intt_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        ((LanternObject<torch::Tensor>*)min)->get(), ((LanternObject<torch::Tensor>*)min_indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().min(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor_dimname_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        ((LanternObject<torch::Tensor>*)min)->get(), ((LanternObject<torch::Tensor>*)min_indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::amin(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amin_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().amin(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::amin_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_weights_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward_weights(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_double(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm_backward(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_var).get())->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_bias_tensor(void* grad_output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_backward_bias(
        ((LanternObject<torch::Tensor>*)grad_output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_transpose_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_input_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_transpose_backward_input(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_convolution_transpose_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_depthwise_convolution(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_input_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_depthwise_convolution_backward_input(
        ((LanternObject<std::vector<int64_t>>*)self_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_depthwise_convolution_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_weight_intarrayref_tensor_tensor_intarrayref_intarrayref_intarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::miopen_depthwise_convolution_backward_weight(
        ((LanternObject<std::vector<int64_t>>*)weight_size)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_tensor_tensorlist_intt_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor(void* input, void* weight, void* weight_stride0, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(cx).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(dropout_state).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_backward_tensor_tensorlist_intt_tensor_tensor_tensor_tensor_tensor_tensor_tensor_intt_intt_intt_bool_double_bool_bool_intarrayref_tensor_tensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn_backward(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<torch::Tensor>*)weight_buf)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(cx).get())->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_output).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_cy).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<std::vector<int64_t>>*)batch_sizes)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(dropout_state).get())->get(), ((LanternObject<torch::Tensor>*)reserve)->get(), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mm(
        ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_out_tensor_tensor_tensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_mm_tensor_tensor(void* sparse, void* dense)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_mm(
        ((LanternObject<torch::Tensor>*)sparse)->get(), ((LanternObject<torch::Tensor>*)dense)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().mode(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_tensor_dimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().mode(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mul(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mul(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mul_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mul_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mul(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mul(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mul_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multiply(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().multiply(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().multiply_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multiply_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multiply(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().multiply(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().multiply_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_tensor_tensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mv(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mv_tensor_tensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mv(
        ((LanternObject<torch::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_out_tensor_tensor_tensor(void* out, void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mv_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mvlgamma_tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mvlgamma(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma_tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mvlgamma(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma__tensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().mvlgamma_(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_copy_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().narrow_copy(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::narrow(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_tensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().narrow(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_tensor_intt_tensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::narrow(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_tensor_intt_tensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().narrow(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_out_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_double(void* out, void* save_mean, void* save_invstd, void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)save_mean)->get(), ((LanternObject<torch::Tensor>*)save_invstd)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<torch::Tensor>*)running_mean)->get(), ((LanternObject<torch::Tensor>*)running_var)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_stats_tensor_double(void* input, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_stats(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_tensor_tensor_tensor_tensor_tensor_double(void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::batch_norm_elemt(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_out_tensor_tensor_tensor_tensor_tensor_tensor_double(void* out, void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::batch_norm_elemt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_tensor_tensor_tensor_tensor_tensor_double_double_intt(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* count)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)count)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_with_counts_tensor_tensor_tensor_tensor_tensor_double_double_tensor(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats_with_counts(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<torch::Tensor>*)counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_bool_double_stdarraybool(void* grad_out, void* input, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_invstd, void* train, void* eps, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_backward(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(save_invstd).get())->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_reduce_tensor_tensor_tensor_tensor_tensor_bool_bool_bool(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* input_g, void* weight_g, void* bias_g)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_backward_reduce(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<bool>*)input_g)->get(), ((LanternObject<bool>*)weight_g)->get(), ((LanternObject<bool>*)bias_g)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_elemt_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* mean_dy, void* mean_dy_xmu)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::batch_norm_backward_elemt(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<torch::Tensor>*)mean_dy)->get(), ((LanternObject<torch::Tensor>*)mean_dy_xmu)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_update_stats_tensor_tensor_tensor_double(void* input, void* running_mean, void* running_var, void* momentum)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_update_stats(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get())));
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
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_tensor_tensor_tensor_intarrayref_stdarraybool(void* input, void* grad_output, void* weight, void* padding, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_nnpack_spatial_convolution_backward(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_input_tensor_tensor_tensor_intarrayref(void* input, void* grad_output, void* weight, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution_backward_input(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_weight_tensor_intarrayref_tensor_intarrayref(void* input, void* weightsize, void* grad_output, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_nnpack_spatial_convolution_backward_weight(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<int64_t>>*)weightsize)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ones(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ones(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ones_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ones_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pairwise_distance_tensor_tensor_double_double_bool(void* x1, void* x2, void* p, void* eps, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pairwise_distance(
        ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cdist_tensor_tensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cdist(
        ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__euclidean_dist_tensor_tensor(void* x1, void* x2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_euclidean_dist(
        ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_forward_tensor_tensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cdist_forward(
        ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_backward_tensor_tensor_tensor_double_tensor(void* grad, void* x1, void* x2, void* p, void* cdist)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cdist_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<torch::Tensor>*)cdist)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pdist_tensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pdist(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_forward_tensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_pdist_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_backward_tensor_tensor_double_tensor(void* grad, void* self, void* p, void* pdist)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_pdist_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<torch::Tensor>*)pdist)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_similarity_tensor_tensor_intt_double(void* x1, void* x2, void* dim, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cosine_similarity(
        ((LanternObject<torch::Tensor>*)x1)->get(), ((LanternObject<torch::Tensor>*)x2)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_permute_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().permute(
        ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::movedim(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_tensor_intarrayref_intarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().movedim(
        ((LanternObject<std::vector<int64_t>>*)source)->get(), ((LanternObject<std::vector<int64_t>>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::movedim(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_tensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().movedim(
        ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_numpy_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().numpy_T(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pixel_shuffle_tensor_intt(void* self, void* upscale_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pixel_shuffle(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)upscale_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_channel_shuffle_tensor_intt(void* self, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::channel_shuffle(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_pinned_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_pinned(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pin_memory_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pin_memory(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pinverse_tensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pinverse(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pinverse_tensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pinverse(
        ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_nll_loss_tensor_tensor_bool_bool_double_intt(void* input, void* target, void* log_input, void* full, void* eps, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::poisson_nll_loss(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<bool>*)log_input)->get(), ((LanternObject<bool>*)full)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rad2deg(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rad2deg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rad2deg_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rad2deg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rad2deg_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::deg2rad(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().deg2rad(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::deg2rad_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().deg2rad_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::deg2rad_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scalar_tensor_scalar_tensoroptions(void* s, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scalar_tensor(
        ((LanternObject<torch::Scalar>*)s)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rand_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intarrayref_tensoroptions(void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intarrayref_generator_tensoroptions(void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_intarrayref_tensoroptions(void* low, void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_intarrayref_generator_tensoroptions(void* low, void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intarrayref(void* out, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intarrayref_generator(void* out, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intt_intarrayref(void* out, void* low, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_tensor_intt_intt_intarrayref_generator(void* out, void* low, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_tensor_intt_tensoroptions_memoryformat(void* self, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_tensor_intt_intt_tensoroptions_memoryformat(void* self, void* low, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randint_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_generator_tensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_intarrayref_generator_dimnamelist_tensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_tensor_intarrayref_generator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randn_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_tensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_generator_tensoroptions(void* n, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_tensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randperm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_tensor_intt_generator(void* out, void* n, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::randperm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_scalar_scalar_scalar_tensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_scalar_scalar_tensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::range(
        ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_out_tensor_scalar_scalar_scalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::range_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)start)->get(), ((LanternObject<torch::Scalar>*)end)->get(), ((LanternObject<torch::Scalar>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reciprocal(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().reciprocal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reciprocal_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().reciprocal_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reciprocal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::neg(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().neg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::neg_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().neg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::neg_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::negative(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().negative(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::negative_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().negative_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::negative_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_tensor_intarrayref(void* self, void* repeats)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().repeat(
        ((LanternObject<std::vector<int64_t>>*)repeats)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor(void* repeats)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::repeat_interleave(
        ((LanternObject<torch::Tensor>*)repeats)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor_tensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::repeat_interleave(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_tensor_tensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().repeat_interleave(
        ((LanternObject<torch::Tensor>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_tensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::repeat_interleave(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_tensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().repeat_interleave(
        ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reshape(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().reshape(
        ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_reshape_tensor_intarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_mkldnn_reshape(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().reshape_as(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_round_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::round(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().round(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::round_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().round_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::round_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu__tensor_scalar_scalar_bool_generator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_relu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::relu(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().relu(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_relu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::relu_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().relu_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_tensor_tensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prelu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_tensor_tensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().prelu(
        ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_backward_tensor_tensor_tensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::prelu_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_backward_tensor_tensor_tensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)grad_output)->get().prelu_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gelu(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_backward_tensor_tensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gelu_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_infinitely_differentiable_gelu_backward_tensor_tensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::infinitely_differentiable_gelu_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardshrink(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().hardshrink(
        ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_backward_tensor_tensor_scalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardshrink_backward(
        ((LanternObject<torch::Tensor>*)grad_out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_backward_tensor_tensor_scalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)grad_out)->get().hardshrink_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rsqrt(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rsqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rsqrt_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rsqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rsqrt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_tensor_dimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::select(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_tensor_dimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().select(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_tensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::select(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_tensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().select(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_backward_tensor_intarrayref_intt_intt(void* grad, void* input_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::select_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_selu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::selu(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_selu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::selu_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_celu_tensor_scalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::celu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_celu__tensor_scalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::celu_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::silu(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::silu_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::silu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::silu_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sigmoid(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sigmoid(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sigmoid_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sigmoid_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sigmoid_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logit(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit_tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logit(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit__tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logit_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit__tensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().logit_(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_out_tensor_tensor_double(void* out, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logit_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sin(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sin_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sin_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sinh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sinh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sinh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_detach_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::detach(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().detach(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_detach__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::detach_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().detach_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_size_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_size_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().size(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_size_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_size_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().size(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_tensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slice(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slice_tensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().slice(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_backward_tensor_intarrayref_intt_intt_intt_intt(void* grad, void* input_sizes, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slice_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slogdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slogdet(
        ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slogdet_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().slogdet(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_smm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::smm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_smm_tensor_tensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().smm(
        ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().softmax(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_softmax_backward_data(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_split(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().unsafe_split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::split(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_tensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unsafe_split_with_sizes(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().unsafe_split_with_sizes(
        ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::split_with_sizes(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_with_sizes_tensor_intarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().split_with_sizes(
        ((LanternObject<std::vector<int64_t>>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::squeeze(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::squeeze(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::squeeze(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().squeeze_(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sspaddmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sspaddmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sspaddmm(
        ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sspaddmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::stack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::stack_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hstack_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::vstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::vstack_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_tensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dstack(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_out_tensor_tensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dstack_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::stft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(window).get())->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stft_tensor_intt_intt_intt_tensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().stft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(window).get())->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::istft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(window).get())->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_istft_tensor_intt_intt_intt_tensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().istft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(window).get())->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().stride(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().stride(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sum(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sum(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_tensor_dimnamelist_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sum(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_tensor_tensor_dimnamelist_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nansum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nansum(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nansum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_tensor_intarrayref_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nansum(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_out_tensor_tensor_intarrayref_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nansum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_to_size_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sum_to_size(
        ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sqrt(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sqrt_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sqrt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_square_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::square(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().square(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_square__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::square_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().square_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::std(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().std(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::std(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().std(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::std_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::std(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().std(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::std_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().prod(
        ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_intt_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_intt_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().prod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_tensor_tensor_intt_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_tensor_dimname_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_tensor_dimname_bool_scalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().prod(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_tensor_tensor_dimname_bool_scalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::prod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::t(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().t(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().t_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tan(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tan_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tan_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tanh(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tanh_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tanh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensordot_tensor_tensor_intarrayref_intarrayref(void* self, void* other, void* dims_self, void* dims_other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tensordot(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<std::vector<int64_t>>*)dims_self)->get(), ((LanternObject<std::vector<int64_t>>*)dims_other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_tensor_scalar_scalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::threshold(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold__tensor_scalar_scalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_out_tensor_tensor_scalar_scalar(void* out, void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_backward_tensor_tensor_scalar(void* grad_output, void* self, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::threshold_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().transpose(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_tensor_dimname_dimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim0)->get(), ((LanternObject<torch::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_tensor_dimname_dimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().transpose(
        ((LanternObject<torch::Dimname>*)dim0)->get(), ((LanternObject<torch::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose_tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_mkldnn_transpose(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().transpose_(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose__tensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_mkldnn_transpose_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_one_hot_tensor_intt(void* self, void* num_classes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::one_hot(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_classes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flip_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flip(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flip_tensor_intarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flip(
        ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fliplr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fliplr(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fliplr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fliplr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_flipud_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::flipud(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flipud_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().flipud(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_roll_tensor_intarrayref_intarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::roll(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)shifts)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_roll_tensor_intarrayref_intarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().roll(
        ((LanternObject<std::vector<int64_t>>*)shifts)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rot90_tensor_intt_intarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rot90(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rot90_tensor_intt_intarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().rot90(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<std::vector<int64_t>>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_tensor_tensor_intt(void* y, void* x, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trapz(
        ((LanternObject<torch::Tensor>*)y)->get(), ((LanternObject<torch::Tensor>*)x)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_tensor_double_intt(void* y, void* dx, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trapz(
        ((LanternObject<torch::Tensor>*)y)->get(), ((LanternObject<double>*)dx)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__trilinear_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intt(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_trilinear(
        ((LanternObject<torch::Tensor>*)i1)->get(), ((LanternObject<torch::Tensor>*)i2)->get(), ((LanternObject<torch::Tensor>*)i3)->get(), ((LanternObject<std::vector<int64_t>>*)expand1)->get(), ((LanternObject<std::vector<int64_t>>*)expand2)->get(), ((LanternObject<std::vector<int64_t>>*)expand3)->get(), ((LanternObject<std::vector<int64_t>>*)sumdim)->get(), ((LanternObject<int64_t>*)unroll_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triplet_margin_loss_tensor_tensor_tensor_double_double_double_bool_intt(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::triplet_margin_loss(
        ((LanternObject<torch::Tensor>*)anchor)->get(), ((LanternObject<torch::Tensor>*)positive)->get(), ((LanternObject<torch::Tensor>*)negative)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)swap)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trunc(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().trunc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trunc_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().trunc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trunc_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fix(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fix(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fix_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fix_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fix_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_type_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().type_as(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__has_compatible_shallow_copy_type_tensor_tensor(void* self, void* from)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_has_compatible_shallow_copy_type(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)from)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__unique_tensor_bool_bool(void* self, void* sorted, void* return_inverse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_tensor_intt_bool_bool_bool(void* self, void* dim, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_consecutive_tensor_bool_bool_intt(void* self, void* return_inverse, void* return_counts, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_consecutive(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_consecutive_tensor_intt_bool_bool(void* self, void* dim, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim_consecutive(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unique2_tensor_bool_bool_bool(void* self, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique2(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unsafe_view_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_unsafe_view(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsqueeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::unsqueeze(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().unsqueeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze__tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().unsqueeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vander_tensor_intt_bool(void* x, void* N, void* increasing)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::vander(
        ((LanternObject<torch::Tensor>*)x)->get(), ((LanternObject<c10::optional<int64_t>>*)N)->get(), ((LanternObject<bool>*)increasing)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::var(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().var(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::var(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().var(
        ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_intarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::var_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::var(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().var(
        ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_tensor_tensor_dimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::var_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_intarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_tensor_dimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_as_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().view_as(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::where(
        ((LanternObject<torch::Tensor>*)condition)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)condition)->get().where(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_scalar_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::where(
        ((LanternObject<torch::Tensor>*)condition)->get(), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_tensor_scalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::where(
        ((LanternObject<torch::Tensor>*)condition)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor_scalar_scalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::where(
        ((LanternObject<torch::Tensor>*)condition)->get(), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_tensor(void* condition)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::where(
        ((LanternObject<torch::Tensor>*)condition)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__s_where_tensor_tensor_tensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_s_where(
        ((LanternObject<torch::Tensor>*)condition)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_except_dim_tensor_intt_intt(void* v, void* pow, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm_except_dim(
        ((LanternObject<torch::Tensor>*)v)->get(), ((LanternObject<int64_t>*)pow)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_tensor_tensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_weight_norm(
        ((LanternObject<torch::Tensor>*)v)->get(), ((LanternObject<torch::Tensor>*)g)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_tensor_tensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface(
        ((LanternObject<torch::Tensor>*)v)->get(), ((LanternObject<torch::Tensor>*)g)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface_backward(
        ((LanternObject<torch::Tensor>*)grad_w)->get(), ((LanternObject<torch::Tensor>*)saved_v)->get(), ((LanternObject<torch::Tensor>*)saved_g)->get(), ((LanternObject<torch::Tensor>*)saved_norms)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_differentiable_backward_tensor_tensor_tensor_tensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_differentiable_backward(
        ((LanternObject<torch::Tensor>*)grad_w)->get(), ((LanternObject<torch::Tensor>*)saved_v)->get(), ((LanternObject<torch::Tensor>*)saved_g)->get(), ((LanternObject<torch::Tensor>*)saved_norms)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_intarrayref_dimnamelist_tensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternPtr<std::vector<torch::Dimname>>*)names)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::zeros(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_out_tensor_intarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::zeros_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_like_tensor_tensoroptions_memoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::zeros_like(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_grad_tensor_tensor(void* self, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_standard_gamma_grad(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_standard_gamma(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dirichlet_grad_tensor_tensor_tensor(void* x, void* alpha, void* total)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_dirichlet_grad(
        ((LanternObject<torch::Tensor>*)x)->get(), ((LanternObject<torch::Tensor>*)alpha)->get(), ((LanternObject<torch::Tensor>*)total)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sample_dirichlet_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sample_dirichlet(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::poisson(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binomial_tensor_tensor_generator(void* count, void* prob, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::binomial(
        ((LanternObject<torch::Tensor>*)count)->get(), ((LanternObject<torch::Tensor>*)prob)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::native_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::native_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_scalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_intarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_tensor_intarrayref_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_backward_tensor_tensor_intarrayref(void* grad, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_sum_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_softmax_backward_data(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_intt_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_dimname_scalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_tensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_backward_data_tensor_tensor_intt_tensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_log_softmax_backward_data(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_scalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_scalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_intarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_intarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_intarrayref_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_dimnamelist_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_dimnamelist_bool_scalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_tensor_scalar_dimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_tensor_scalar_dimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool_scalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_tensor_tensor_scalar_dimnamelist_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(p).get())->get(), ((LanternPtr<std::vector<torch::Dimname>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::frobenius_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_tensor_bool(void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_tensor_tensor_bool(void* out, void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_tensor_intarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_tensor_tensor_intarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clone_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::clone(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clone_tensor_memoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().clone(
        ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_resize_as__tensor_tensor_memoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::resize_as_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)the_template)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize_as__tensor_tensor_memoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().resize_as_(
        ((LanternObject<torch::Tensor>*)the_template)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zero__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::zero_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_zero__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().zero_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sub_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sub(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sub(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sub_(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sub(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sub(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sub_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_out_tensor_tensor_tensor_scalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::subtract_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::subtract(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().subtract(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().subtract_(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::subtract(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().subtract(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().subtract_(
        ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rsub(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_out_tensor_tensor_tensor(void* out, void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::heaviside_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::heaviside(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside_tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().heaviside(
        ((LanternObject<torch::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside__tensor_tensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().heaviside_(
        ((LanternObject<torch::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_tensor_scalar_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rsub(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* sparse, void* dense, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_addmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)sparse)->get(), ((LanternObject<torch::Tensor>*)dense)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm_tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addmm(
        ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm__tensor_tensor_tensor_scalar_scalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addmm_(
        ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_intarrayref_tensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_tensor_tensor_tensoroptions(void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_unsafe_tensor_tensor_intarrayref_tensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_coo_tensor_unsafe(
        ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__validate_sparse_coo_tensor_args_tensor_tensor_intarrayref(void* indices, void* values, void* size)
{
  LANTERN_FUNCTION_START
    torch::_validate_sparse_coo_tensor_args(((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_intt_intt_intarrayref_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_coo_tensor_with_dims(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_intarrayref_tensor_tensor_tensoroptions(void* sparse_dim, void* dense_dim, void* size, void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_sparse_coo_tensor_with_dims_and_tensors(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sparse_resize_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize_and_clear__tensor_intarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sparse_resize_and_clear_(
        ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_mask_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sparse_mask(
        ((LanternObject<torch::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_dense_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to_dense(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_to_dense_backward_tensor_tensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::to_dense_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_dim_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().sparse_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimi_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get()._dimI(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dense_dim_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().dense_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get()._dimV(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__nnz_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get()._nnz(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_coalesce_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().coalesce(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_coalesced_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_coalesced(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get()._indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__values_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get()._values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__coalesced__tensor_bool(void* self, void* coalesced)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get()._coalesced_(
        ((LanternObject<bool>*)coalesced)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_indices_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_values_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_out_tensor_tensor_tensor(void* out, void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hspmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_tensor_tensor(void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hspmm(
        ((LanternObject<torch::Tensor>*)mat1)->get(), ((LanternObject<torch::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copy_sparse_to_sparse__tensor_tensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::copy_sparse_to_sparse_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)src)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unbind(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().unbind(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::unbind(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_tensor_dimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().unbind(
        ((LanternObject<torch::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_tensor_intt(void* self, void* sparse_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to_sparse(
        ((LanternObject<int64_t>*)sparse_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to_sparse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_mkldnn_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to_mkldnn(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv2d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_reorder_conv2d_weight(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv3d_weight_tensor_intarrayref_intarrayref_intarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_reorder_conv3d_weight(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_to_mkldnn_backward_tensor_tensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::to_mkldnn_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_tensor_double_intt_scalartype(void* self, void* scale, void* zero_point, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantize_per_tensor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_tensorlist_tensor_tensor_scalartype(void* tensors, void* scales, void* zero_points, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::quantize_per_tensor(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<torch::Tensor>*)scales)->get(), ((LanternObject<torch::Tensor>*)zero_points)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_channel_tensor_tensor_tensor_intt_scalartype(void* self, void* scales, void* zero_points, void* axis, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantize_per_channel(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scales)->get(), ((LanternObject<torch::Tensor>*)zero_points)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dequantize_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dequantize(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dequantize_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().dequantize(
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
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_scale_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<double>(((LanternObject<torch::Tensor>*)self)->get().q_scale(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_zero_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_zero_point(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_zero_point_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().q_zero_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_scales_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::q_per_channel_scales(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_scales_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().q_per_channel_scales(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_zero_points_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::q_per_channel_zero_points(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_zero_points_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().q_per_channel_zero_points(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_axis_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_per_channel_axis(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_axis_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<torch::Tensor>*)self)->get().q_per_channel_axis(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_int_repr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::int_repr(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_int_repr_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().int_repr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_tensor_quantized_tensor_tensor_double_intt(void* self, void* scale, void* zero_point)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_make_per_tensor_quantized_tensor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_channel_quantized_tensor_tensor_tensor_tensor_intt(void* self, void* scale, void* zero_point, void* axis)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_make_per_channel_quantized_tensor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qscheme_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::QScheme>(((LanternObject<torch::Tensor>*)self)->get().qscheme(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_tensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fake_quantize_per_tensor_affine(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_backward_tensor_tensor_double_intt_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fake_quantize_per_tensor_affine_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_tensor_tensor_tensor_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_fake_quantize_learnable_per_tensor_affine(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_backward_tensor_tensor_tensor_tensor_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_tensor_affine_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fake_quantize_per_channel_affine(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fake_quantize_per_channel_affine_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_tensor_tensor_tensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_fake_quantize_learnable_per_channel_affine(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_backward_tensor_tensor_tensor_tensor_intt_intt_intt(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_channel_affine_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)scale)->get(), ((LanternObject<torch::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__choose_qparams_per_tensor_tensor_bool(void* self, void* reduce_range)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_choose_qparams_per_tensor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)reduce_range)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__saturate_weight_to_fp16_tensor(void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_saturate_weight_to_fp16(
        ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_choose_qparams_optimized_tensor_intt_intt_double_intt(void* input, void* numel, void* n_bins, void* ratio, void* bit_width)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::choose_qparams_optimized(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<int64_t>*)numel)->get(), ((LanternObject<int64_t>*)n_bins)->get(), ((LanternObject<double>*)ratio)->get(), ((LanternObject<int64_t>*)bit_width)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_tensoroptions_bool_bool_memoryformat(void* self, void* options, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to(
        ((LanternObject<torch::TensorOptions>*)options)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_device_scalartype_bool_bool_memoryformat(void* self, void* device, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to(
        ((LanternPtr<torch::Device>*)device)->get(), ((LanternObject<torch::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_scalartype_bool_bool_memoryformat(void* self, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to(
        ((LanternObject<torch::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_tensor_tensor_bool_bool_memoryformat(void* self, void* other, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().to(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<torch::MemoryFormat>>*)optional<torch::MemoryFormat>(memory_format).get())->get()));
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
    return (void *) new LanternObject<torch::Tensor>(torch::cartesian_prod(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_combinations_tensor_intt_bool(void* self, void* r, void* with_replacement)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::combinations(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)r)->get(), ((LanternObject<bool>*)with_replacement)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_item_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Scalar>(((LanternObject<torch::Tensor>*)self)->get().item(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_tensor_tensor(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Tensor>*)tensor)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_tensor_scalar(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Tensor>*)tensor)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_scalar_tensor(void* scalar, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::ScalarType>(torch::result_type(
        ((LanternObject<torch::Scalar>*)scalar)->get(), ((LanternObject<torch::Tensor>*)tensor)->get()));
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
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* cx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell(
        ((LanternObject<torch::Tensor>*)input_gates)->get(), ((LanternObject<torch::Tensor>*)hidden_gates)->get(), ((LanternObject<torch::Tensor>*)cx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_bool(void* grad_hy, void* grad_cy, void* cx, void* cy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_cy).get())->get(), ((LanternObject<torch::Tensor>*)cx)->get(), ((LanternObject<torch::Tensor>*)cy)->get(), ((LanternObject<torch::Tensor>*)workspace)->get(), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_lstm_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* grad_cy, void* input_gates, void* hidden_gates, void* input_bias, void* hidden_bias, void* cx, void* cy)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_lstm_cell_backward(
        ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(grad_cy).get())->get(), ((LanternObject<torch::Tensor>*)input_gates)->get(), ((LanternObject<torch::Tensor>*)hidden_gates)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(hidden_bias).get())->get(), ((LanternObject<torch::Tensor>*)cx)->get(), ((LanternObject<torch::Tensor>*)cy)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_tensor_tensor_tensor_tensor_tensor(void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell(
        ((LanternObject<torch::Tensor>*)input_gates)->get(), ((LanternObject<torch::Tensor>*)hidden_gates)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_backward_tensor_tensor_bool(void* grad_hy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell_backward(
        ((LanternObject<torch::Tensor>*)grad_hy)->get(), ((LanternObject<torch::Tensor>*)workspace)->get(), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_gru_cell_backward_tensor_tensor_tensor_tensor_tensor_tensor(void* grad_hy, void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_gru_cell_backward(
        ((LanternObject<torch::Tensor>*)grad_hy)->get(), ((LanternObject<torch::Tensor>*)input_gates)->get(), ((LanternObject<torch::Tensor>*)hidden_gates)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_tensor_tensor_tensorlist_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        ((LanternObject<torch::Tensor>*)data)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        ((LanternObject<torch::Tensor>*)data)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        ((LanternObject<torch::Tensor>*)data)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_tensor_tensor_tensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_tensor_tensor_tensor_tensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        ((LanternObject<torch::Tensor>*)data)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<std::vector<torch::Tensor>>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_hh).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gru_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rnn_tanh_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rnn_relu_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_lstm_cell_tensor_tensorlist_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::quantized_lstm_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<std::vector<torch::Tensor>>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<torch::Tensor>*)b_ih)->get(), ((LanternObject<torch::Tensor>*)b_hh)->get(), ((LanternObject<torch::Tensor>*)packed_ih)->get(), ((LanternObject<torch::Tensor>*)packed_hh)->get(), ((LanternObject<torch::Tensor>*)col_offsets_ih)->get(), ((LanternObject<torch::Tensor>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_gru_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_gru_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<torch::Tensor>*)b_ih)->get(), ((LanternObject<torch::Tensor>*)b_hh)->get(), ((LanternObject<torch::Tensor>*)packed_ih)->get(), ((LanternObject<torch::Tensor>*)packed_hh)->get(), ((LanternObject<torch::Tensor>*)col_offsets_ih)->get(), ((LanternObject<torch::Tensor>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_relu_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_rnn_relu_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<torch::Tensor>*)b_ih)->get(), ((LanternObject<torch::Tensor>*)b_hh)->get(), ((LanternObject<torch::Tensor>*)packed_ih)->get(), ((LanternObject<torch::Tensor>*)packed_hh)->get(), ((LanternObject<torch::Tensor>*)col_offsets_ih)->get(), ((LanternObject<torch::Tensor>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_tanh_cell_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_tensor_scalar_scalar_scalar_scalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantized_rnn_tanh_cell(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)hx)->get(), ((LanternObject<torch::Tensor>*)w_ih)->get(), ((LanternObject<torch::Tensor>*)w_hh)->get(), ((LanternObject<torch::Tensor>*)b_ih)->get(), ((LanternObject<torch::Tensor>*)b_hh)->get(), ((LanternObject<torch::Tensor>*)packed_ih)->get(), ((LanternObject<torch::Tensor>*)packed_hh)->get(), ((LanternObject<torch::Tensor>*)col_offsets_ih)->get(), ((LanternObject<torch::Tensor>*)col_offsets_hh)->get(), ((LanternObject<torch::Scalar>*)scale_ih)->get(), ((LanternObject<torch::Scalar>*)scale_hh)->get(), ((LanternObject<torch::Scalar>*)zero_point_ih)->get(), ((LanternObject<torch::Scalar>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_tensor_tensor_bool(void* input, void* lengths, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pack_padded_sequence(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)lengths)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_backward_tensor_intarrayref_tensor_bool(void* grad, void* input_size, void* batch_sizes, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_pack_padded_sequence_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<bool>*)batch_first)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pad_packed_sequence_tensor_tensor_bool_scalar_intt(void* data, void* batch_sizes, void* batch_first, void* padding_value, void* total_length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pad_packed_sequence(
        ((LanternObject<torch::Tensor>*)data)->get(), ((LanternObject<torch::Tensor>*)batch_sizes)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<torch::Scalar>*)padding_value)->get(), ((LanternObject<int64_t>*)total_length)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_storage(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().set_(
        ((LanternObject<torch::Storage>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_storage_intt_intarrayref_intarrayref(void* self, void* source, void* storage_offset, void* size, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().set_(
        ((LanternObject<torch::Storage>*)source)->get(), ((LanternObject<int64_t>*)storage_offset)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor_tensor(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().set_(
        ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().set_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set_quantizer__tensor_constquantizerptr(void* self, void* quantizer)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().set_quantizer_(
        ((LanternObject<torch::ConstQuantizerPtr>*)quantizer)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_set_to_tensor_tensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().is_set_to(
        ((LanternObject<torch::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_fill_(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_tensor_tensor_scalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_fill(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_fill_(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_tensor_tensor_tensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_fill(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter__tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_scatter_(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_scatter_tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_scatter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter_tensor_tensor_tensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_scatter(
        ((LanternObject<torch::Tensor>*)mask)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_tensor_intarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().view(
        ((LanternObject<std::vector<int64_t>>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_put__tensor_tensor_tensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().put_(
        ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_add_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_add(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_add(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill_(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill_(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_fill(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_fill(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_intt_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_tensor_dimname_tensor_scalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_tensor_stdstring(void* self, void* dim, void* index, void* src, void* reduce)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get(), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__tensor_intt_tensor_scalar_stdstring(void* self, void* dim, void* index, void* value, void* reduce)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Scalar>*)value)->get(), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_add_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter_add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_add(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::scatter_add(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_tensor_dimname_tensor_tensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().scatter_add(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().eq_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().eq_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_and_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_and_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_and(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_and(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_and(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_and(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_and_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_and_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___and___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__and__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__and__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___and___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__and__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__and__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__iand__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__iand__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_or_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_or_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_or(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_or(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_or(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_or(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_or_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_or_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___or___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__or__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__or__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___or___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__or__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__or__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ior__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ior__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_xor_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_xor_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_xor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_xor(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bitwise_xor(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_xor(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_xor_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().bitwise_xor_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__xor__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__xor__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__xor__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__xor__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ixor__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ixor__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__lshift__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__lshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__lshift__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__lshift__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ilshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__ilshift__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__rshift__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__rshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::__rshift__(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__rshift__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__irshift__(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().__irshift__(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lgamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atan2_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril__tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tril_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu__tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().triu_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().digamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_polygamma__tensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().polygamma_(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm__tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().renorm_(
        ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pow_(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pow_(
        ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lerp_(
        ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lerp_(
        ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fmod_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fmod_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().remainder_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().remainder_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm__tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addbmm_(
        ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_out_tensor_tensor_tensor_tensor_scalar_scalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addbmm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addbmm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm_tensor_tensor_tensor_scalar_scalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addbmm(
        ((LanternObject<torch::Tensor>*)batch1)->get(), ((LanternObject<torch::Tensor>*)batch2)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addcdiv_(
        ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_intt_intt_generator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().random_(
        ((LanternObject<int64_t>*)from)->get(), ((LanternObject<c10::optional<int64_t>>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_intt_generator(void* self, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().random_(
        ((LanternObject<int64_t>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__tensor_generator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().random_(
        ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_uniform__tensor_double_double_generator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().uniform_(
        ((LanternObject<double>*)from)->get(), ((LanternObject<double>*)to)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cauchy__tensor_double_double_generator(void* self, void* median, void* sigma, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cauchy_(
        ((LanternObject<double>*)median)->get(), ((LanternObject<double>*)sigma)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().log_normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exponential__tensor_double_generator(void* self, void* lambd, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().exponential_(
        ((LanternObject<double>*)lambd)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geometric__tensor_double_generator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().geometric_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diag_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diag(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().diag(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_backward_tensor_intarrayref_intt(void* grad, void* input_sizes, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::diag_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_out_tensor_tensor_tensor_intt(void* out, void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cross_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_tensor_tensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cross(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cross_tensor_tensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cross(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::triu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::triu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().triu(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_out_tensor_tensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tril_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tril(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril_tensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().tril(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tril_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_indices_intt_intt_intt_tensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::triu_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<torch::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trace(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trace_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().trace(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_backward_tensor_intarrayref(void* grad, void* sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::trace_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ne_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ne(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ne(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ne_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ne(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ne(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ne_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ne_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::not_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::not_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().not_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::not_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::not_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().not_equal(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().not_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().not_equal_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eq_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eq(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().eq(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eq_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::eq(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().eq(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ge_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ge(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ge(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ge_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ge(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ge(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ge_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ge_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_equal(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_equal_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::le_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::le(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().le(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::le_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::le(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().le(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().le_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().le_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_equal(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_equal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_equal(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_equal_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_equal_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gt(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gt(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gt(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gt(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gt_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gt_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::greater(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().greater_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lt(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lt(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lt_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lt(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lt(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lt_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lt_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::less(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().less_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_out_tensor_tensor_tensor(void* out, void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::take_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_tensor_tensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::take(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_take_tensor_tensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().take(
        ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_backward_tensor_tensor_tensor(void* grad, void* input, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::take_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_tensor_tensor_intt_tensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_select_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_tensor_intt_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_select(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_tensor_intt_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_select(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_tensor_tensor_dimname_tensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_select_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_tensor_dimname_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_select(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_tensor_dimname_tensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().index_select(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_backward_tensor_intarrayref_intt_tensor(void* grad, void* self_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::index_select_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<std::vector<int64_t>>*)self_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_out_tensor_tensor_tensor(void* out, void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_select_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_select(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_select_tensor_tensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().masked_select(
        ((LanternObject<torch::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_backward_tensor_tensor_tensor(void* grad, void* input, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::masked_select_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<torch::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nonzero_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nonzero(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_numpy_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::nonzero_numpy(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_numpy_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(((LanternObject<torch::Tensor>*)self)->get().nonzero_numpy(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_tensor_tensor_intt_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gather_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_tensor_intt_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gather(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_tensor_intt_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gather(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_backward_tensor_tensor_intt_tensor_bool(void* grad, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gather_backward(
        ((LanternObject<torch::Tensor>*)grad)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_tensor_tensor_dimname_tensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gather_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_tensor_dimname_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::gather(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_tensor_dimname_tensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().gather(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__gather_sparse_backward_tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_gather_sparse_backward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addcmul_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addcmul(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addcmul(
        ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul__tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addcmul_(
        ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_out_tensor_tensor_tensor_tensor_scalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addcdiv_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::addcdiv(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv_tensor_tensor_tensor_scalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().addcdiv(
        ((LanternObject<torch::Tensor>*)tensor1)->get(), ((LanternObject<torch::Tensor>*)tensor2)->get(), ((LanternObject<torch::Scalar>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_out_tensor_tensor_tensor_tensor(void* X, void* qr, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq_out(
        ((LanternObject<torch::Tensor>*)X)->get(), ((LanternObject<torch::Tensor>*)qr)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lstsq_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().lstsq(
        ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_out_tensor_tensor_tensor_tensor_bool_bool_bool(void* X, void* M, void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve_out(
        ((LanternObject<torch::Tensor>*)X)->get(), ((LanternObject<torch::Tensor>*)M)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triangular_solve_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().triangular_solve(
        ((LanternObject<torch::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__triangular_solve_helper_tensor_tensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_triangular_solve_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_out_tensor_tensor_tensor_bool_bool(void* e, void* V, void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig_out(
        ((LanternObject<torch::Tensor>*)e)->get(), ((LanternObject<torch::Tensor>*)V)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_symeig_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().symeig(
        ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__symeig_helper_tensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_symeig_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_out_tensor_tensor_tensor_bool(void* e, void* v, void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig_out(
        ((LanternObject<torch::Tensor>*)e)->get(), ((LanternObject<torch::Tensor>*)v)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_tensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eig_tensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().eig(
        ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_out_tensor_tensor_tensor_tensor_bool_bool(void* U, void* S, void* V, void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd_out(
        ((LanternObject<torch::Tensor>*)U)->get(), ((LanternObject<torch::Tensor>*)S)->get(), ((LanternObject<torch::Tensor>*)V)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_svd_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().svd(
        ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__svd_helper_tensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_svd_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_out_tensor_tensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cholesky(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cholesky_helper_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cholesky_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_out_tensor_tensor_tensor_bool(void* out, void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_solve_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_tensor_tensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_solve(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_solve_tensor_tensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cholesky_solve(
        ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cholesky_solve_helper_tensor_tensor_bool(void* self, void* A, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cholesky_solve_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_solve_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().solve(
        ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_out_tensor_tensor_tensor_tensor(void* solution, void* lu, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve_out(
        ((LanternObject<torch::Tensor>*)solution)->get(), ((LanternObject<torch::Tensor>*)lu)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__solve_helper_tensor_tensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_solve_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_out_tensor_tensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_inverse_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::cholesky_inverse(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_inverse_tensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().cholesky_inverse(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_out_tensor_tensor_tensor_bool(void* Q, void* R, void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr_out(
        ((LanternObject<torch::Tensor>*)Q)->get(), ((LanternObject<torch::Tensor>*)R)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_tensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qr_tensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().qr(
        ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__qr_helper_tensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_qr_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_out_tensor_tensor_tensor(void* a, void* tau, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf_out(
        ((LanternObject<torch::Tensor>*)a)->get(), ((LanternObject<torch::Tensor>*)tau)->get(), ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf(
        ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geqrf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().geqrf(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_out_tensor_tensor_tensor(void* out, void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::orgqr_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_tensor_tensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::orgqr(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_orgqr_tensor_tensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().orgqr(
        ((LanternObject<torch::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_out_tensor_tensor_tensor_tensor_bool_bool(void* out, void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ormqr_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_tensor_tensor_tensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ormqr(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ormqr_tensor_tensor_tensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ormqr(
        ((LanternObject<torch::Tensor>*)input2)->get(), ((LanternObject<torch::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__lu_with_info_tensor_bool_bool(void* self, void* pivot, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_lu_with_info(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)pivot)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_out_tensor_tensor_tensor_tensor(void* out, void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lu_solve_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)LU_data)->get(), ((LanternObject<torch::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lu_solve(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)LU_data)->get(), ((LanternObject<torch::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lu_solve_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lu_solve(
        ((LanternObject<torch::Tensor>*)LU_data)->get(), ((LanternObject<torch::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__lu_solve_helper_tensor_tensor_tensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_lu_solve_helper(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)LU_data)->get(), ((LanternObject<torch::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_out_tensor_tensor_intt_bool_generator(void* out, void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multinomial_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_tensor_intt_bool_generator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multinomial(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multinomial_tensor_intt_bool_generator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().multinomial(
        ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__multinomial_alias_setup_tensor(void* probs)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_multinomial_alias_setup(
        ((LanternObject<torch::Tensor>*)probs)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__multinomial_alias_draw_tensor_tensor_intt_generator(void* J, void* q, void* num_samples, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_multinomial_alias_draw(
        ((LanternObject<torch::Tensor>*)J)->get(), ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lgamma_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lgamma(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lgamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::digamma_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::digamma(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().digamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_polygamma_out_tensor_intt_tensor(void* out, void* n, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::polygamma_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erfinv(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erfinv(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().erfinv_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::erfinv_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::i0(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().i0(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::i0_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().i0_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::i0_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sign(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sign(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().sign_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sign_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::signbit(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_signbit_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().signbit(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::signbit_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dist_tensor_tensor_scalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::dist(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dist_tensor_tensor_scalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().dist(
        ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atan2_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::atan2(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().atan2(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_tensor_tensor_tensor_scalar(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lerp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_tensor_tensor_tensor_tensor(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lerp_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lerp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_tensor_tensor_scalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lerp(
        ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Scalar>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::lerp(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_tensor_tensor_tensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().lerp(
        ((LanternObject<torch::Tensor>*)end)->get(), ((LanternObject<torch::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_out_tensor_tensor_intt_scalar_scalar(void* out, void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::histc_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_tensor_intt_scalar_scalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::histc(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_histc_tensor_intt_scalar_scalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().histc(
        ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<torch::Scalar>*)min)->get(), ((LanternObject<torch::Scalar>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fmod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fmod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fmod(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fmod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fmod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fmod(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hypot_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hypot(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().hypot(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().hypot_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nextafter_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nextafter(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nextafter(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter__tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nextafter_(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_tensor_tensor_scalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::remainder_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::remainder(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_tensor_scalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().remainder(
        ((LanternObject<torch::Scalar>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::remainder_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::remainder(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().remainder(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::min(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().min(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().max(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::maximum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_maximum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().maximum(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::maximum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().max(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::minimum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_minimum_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().minimum(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::minimum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_tensor_tensor_tensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::min_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::min(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().min(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_median_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::median(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().median(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantile_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantile(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().quantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_tensor_tensor_tensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantile_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::quantile(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().quantile(
        ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nanquantile_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nanquantile(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nanquantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_tensor_tensor_tensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nanquantile_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nanquantile(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_tensor_tensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().nanquantile(
        ((LanternObject<torch::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().sort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_tensor_tensor_tensor_dimname_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().sort(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::argsort(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_tensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().argsort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::argsort(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_tensor_dimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().argsort(
        ((LanternObject<torch::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_out_tensor_tensor_tensor_intt_intt_bool_bool(void* values, void* indices, void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_tensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_topk_tensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<torch::Tensor>*)self)->get().topk(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_all_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::all(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().all(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_any_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::any(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().any(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_out_tensor_tensor_scalar_intt_scalar(void* out, void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::renorm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::renorm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm_tensor_scalar_intt_scalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().renorm(
        ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Scalar>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unfold_tensor_intt_intt_intt(void* self, void* dimension, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().unfold(
        ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unfold_backward_tensor_intarrayref_intt_intt_intt(void* grad_in, void* input_sizes, void* dim, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::unfold_backward(
        ((LanternObject<torch::Tensor>*)grad_in)->get(), ((LanternObject<std::vector<int64_t>>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::equal(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_equal_tensor_tensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<torch::Tensor>*)self)->get().equal(
        ((LanternObject<torch::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_tensor_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_tensor_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pow(
        ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_scalar_tensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_scalar_tensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow(
        ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_tensor_tensor_scalar(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::pow(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_tensor_scalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().pow(
        ((LanternObject<torch::Scalar>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_normal__tensor_double_double_generator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_tensor_double_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::normal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_double_tensor_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::normal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<double>*)mean)->get(), ((LanternObject<torch::Tensor>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_tensor_tensor_generator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::normal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)mean)->get(), ((LanternObject<torch::Tensor>*)std)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_tensor_double_double_intarrayref_generator(void* out, void* mean, void* std, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::normal_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<std::vector<int64_t>>*)size)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alias_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::alias(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_alias_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().alias(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__index_copy__tensor_intt_tensor_tensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_index_copy_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<torch::Tensor>*)index)->get(), ((LanternObject<torch::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cumsum(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cumsum_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cumprod(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cumprod_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__var_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_var(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__std_tensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_std(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__amp_non_finite_check_and_unscale__tensor_tensor_tensor(void* self, void* found_inf, void* inv_scale)
{
  LANTERN_FUNCTION_START
    torch::_amp_non_finite_check_and_unscale_(((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)found_inf)->get(), ((LanternObject<torch::Tensor>*)inv_scale)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__amp_update_scale_tensor_tensor_tensor_double_double_intt(void* growth_tracker, void* current_scale, void* found_inf, void* scale_growth_factor, void* scale_backoff_factor, void* growth_interval)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_amp_update_scale(
        ((LanternObject<torch::Tensor>*)growth_tracker)->get(), ((LanternObject<torch::Tensor>*)current_scale)->get(), ((LanternObject<torch::Tensor>*)found_inf)->get(), ((LanternObject<double>*)scale_growth_factor)->get(), ((LanternObject<double>*)scale_backoff_factor)->get(), ((LanternObject<int64_t>*)growth_interval)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_tensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cat(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_out_tensor_tensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_cat_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
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

void* _lantern__foreach_add_scalar_list_tensorlist_arrayrefdouble(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_add_scalar_list(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<std::vector<double>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_scalar_list__tensorlist_arrayrefdouble(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_scalar_list_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<double>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_scalar_list_tensorlist_arrayrefdouble(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_sub_scalar_list(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<std::vector<double>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_scalar_list__tensorlist_arrayrefdouble(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_scalar_list_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<double>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_scalar_list_tensorlist_arrayrefdouble(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_div_scalar_list(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<std::vector<double>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_scalar_list__tensorlist_arrayrefdouble(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_scalar_list_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<double>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_scalar_list_tensorlist_arrayrefdouble(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<torch::Tensor>>(torch::_foreach_mul_scalar_list(
        ((LanternObject<std::vector<torch::Tensor>>*)tensors)->get(), ((LanternObject<std::vector<double>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_scalar_list__tensorlist_arrayrefdouble(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_scalar_list_(((LanternObject<std::vector<torch::Tensor>>*)self)->get(), ((LanternObject<std::vector<double>>*)scalars)->get());
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

void* _lantern__mode_tensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_mode(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__mode_out_tensor_tensor_tensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_mode_out(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_tensor_tensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bucketize(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_out_tensor_tensor_tensor_bool_bool(void* out, void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bucketize_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_scalar_tensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::bucketize(
        ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<torch::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_tensor_tensor_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::searchsorted(
        ((LanternObject<torch::Tensor>*)sorted_sequence)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_out_tensor_tensor_tensor_bool_bool(void* out, void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::searchsorted_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)sorted_sequence)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_tensor_scalar_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::searchsorted(
        ((LanternObject<torch::Tensor>*)sorted_sequence)->get(), ((LanternObject<torch::Scalar>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mse_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mse_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mse_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mse_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::l1_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::l1_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::l1_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::l1_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_out_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* out, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multi_margin_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_tensor_tensor_scalar_scalar_tensor_intt(void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multi_margin_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_out_tensor_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multi_margin_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_tensor_tensor_tensor_scalar_scalar_tensor_intt(void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multi_margin_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Scalar>*)p)->get(), ((LanternObject<torch::Scalar>*)margin)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_out_tensor_tensor_tensor_tensor_intt(void* output, void* is_target, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)is_target)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<torch::Tensor>*)is_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_tensor_tensor_tensor_intt_tensor(void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::multilabel_margin_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<torch::Tensor>*)is_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_out_tensor_tensor_tensor_tensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_out_tensor_tensor_tensor_tensor_tensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_tensor_tensor_tensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_out_tensor_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_tensor_tensor_tensor_tensor_intt_intt_tensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::nll_loss2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<torch::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_out_tensor_tensor_tensor_intt_double(void* out, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::smooth_l1_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_tensor_tensor_intt_double(void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::smooth_l1_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_out_tensor_tensor_tensor_tensor_intt_double(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::smooth_l1_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_tensor_tensor_tensor_intt_double(void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::smooth_l1_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_out_tensor_tensor_tensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::soft_margin_loss_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_tensor_tensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::soft_margin_loss(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_out_tensor_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::soft_margin_loss_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_tensor_tensor_tensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::soft_margin_loss_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_out_tensor_tensor_scalar_scalar_scalar(void* out, void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::elu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::elu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_backward_out_tensor_tensor_scalar_scalar_scalar_tensor(void* grad_input, void* grad_output, void* alpha, void* scale, void* input_scale, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::elu_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_backward_tensor_scalar_scalar_scalar_tensor(void* grad_output, void* alpha, void* scale, void* input_scale, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::elu_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu__tensor_scalar_scalar_scalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::elu_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)alpha)->get(), ((LanternObject<torch::Scalar>*)scale)->get(), ((LanternObject<torch::Scalar>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_out_tensor_tensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::glu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_tensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::glu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_out_tensor_tensor_tensor_intt(void* grad_input, void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::glu_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_tensor_tensor_intt(void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::glu_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardsigmoid_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardsigmoid(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardsigmoid_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardsigmoid_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_out_tensor_tensor_scalar_scalar(void* out, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardtanh_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_tensor_scalar_scalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardtanh(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_out_tensor_tensor_tensor_scalar_scalar(void* grad_input, void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardtanh_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_tensor_tensor_scalar_scalar(void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardtanh_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh__tensor_scalar_scalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardtanh_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)min_val)->get(), ((LanternObject<torch::Scalar>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardswish_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardswish(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish__tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardswish_(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::hardswish_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_out_tensor_tensor_scalar(void* out, void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::leaky_relu_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_tensor_scalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::leaky_relu(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_backward_tensor_tensor_scalar_bool(void* grad_output, void* self, void* negative_slope, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::leaky_relu_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)negative_slope)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu__tensor_scalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::leaky_relu_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_out_tensor_tensor_tensor(void* output, void* buffer, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)buffer)->get(), ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward(
        ((LanternObject<torch::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)buffer)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_tensor_tensor_tensor(void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::log_sigmoid_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)buffer)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_out_tensor_tensor_tensor_scalar_scalar_bool_generator(void* out, void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_with_noise_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)noise)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_with_noise(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)noise)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_backward_tensor_tensor_tensor_scalar_scalar_bool_bool(void* grad_output, void* self, void* noise, void* lower, void* upper, void* training, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_with_noise_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)noise)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise__tensor_tensor_scalar_scalar_bool_generator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::rrelu_with_noise_(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)noise)->get(), ((LanternObject<torch::Scalar>*)lower)->get(), ((LanternObject<torch::Scalar>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<torch::Generator>>*)optional<torch::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_out_tensor_tensor_scalar_scalar(void* out, void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softplus_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_tensor_scalar_scalar(void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softplus(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_out_tensor_tensor_tensor_scalar_scalar_tensor(void* grad_input, void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softplus_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_tensor_tensor_scalar_scalar_tensor(void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softplus_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)beta)->get(), ((LanternObject<torch::Scalar>*)threshold)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_out_tensor_tensor_scalar(void* out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softshrink_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_tensor_scalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softshrink(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_out_tensor_tensor_tensor_scalar(void* grad_input, void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softshrink_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_tensor_tensor_scalar(void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::softshrink_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Scalar>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::mkldnn_adaptive_avg_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_adaptive_avg_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_adaptive_avg_pool2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_out_tensor_tensor_intarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_backward_tensor_tensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_avg_pool3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_max_pool2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_max_pool2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_out_tensor_tensor_tensor_intarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_tensor_intarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_out_tensor_tensor_tensor_tensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_max_pool3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_tensor_tensor_tensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::adaptive_max_pool3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_out_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::avg_pool3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fractional_max_pool2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fractional_max_pool2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_tensor_intarrayref_intarrayref_tensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fractional_max_pool3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_tensor_tensor_intarrayref_intarrayref_tensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fractional_max_pool3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool2d_with_indices_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool2d_with_indices_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool3d_with_indices_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_bool_tensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_pool3d_with_indices_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<torch::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_out_tensor_tensor_tensor_intarrayref(void* out, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_tensor_tensor_intarrayref(void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_out_tensor_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_tensor_tensor_tensor_intarrayref(void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_out_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_tensor_tensor_intarrayref_intarrayref_intarrayref(void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_out_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref(void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::max_unpool3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)indices)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad1d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad1d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::reflection_pad2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad1d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad1d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_out_tensor_tensor_intarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_tensor_intarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_out_tensor_tensor_tensor_intarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_tensor_tensor_intarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::replication_pad3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_tensor_intarrayref_bool_arrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_tensor_intarrayref_arrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d(
        ((LanternObject<torch::Tensor>*)input)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_arrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)scale_factors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_out_tensor_tensor_intarrayref_bool_double(void* out, void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_tensor_intarrayref_bool_double(void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_tensor_intarrayref_intarrayref_bool_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_linear1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_out_tensor_tensor_intarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_tensor_intarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_tensor_intarrayref_intarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bilinear2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_out_tensor_tensor_intarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_tensor_intarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_tensor_intarrayref_intarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_bicubic2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_out_tensor_tensor_intarrayref_bool_double_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_tensor_intarrayref_bool_double_double_double(void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_out_tensor_tensor_intarrayref_intarrayref_bool_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_tensor_intarrayref_intarrayref_bool_double_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_trilinear3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_out_tensor_tensor_intarrayref_double(void* out, void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_tensor_intarrayref_double(void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_out_tensor_tensor_intarrayref_intarrayref_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_tensor_intarrayref_intarrayref_double(void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest1d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_out_tensor_tensor_intarrayref_double_double(void* out, void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_tensor_intarrayref_double_double(void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_tensor_intarrayref_intarrayref_double_double(void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_out_tensor_tensor_intarrayref_double_double_double(void* out, void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_tensor_intarrayref_double_double_double(void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_out_tensor_tensor_intarrayref_intarrayref_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_tensor_intarrayref_intarrayref_double_double_double(void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::upsample_nearest3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sigmoid_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_tensor_tensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::sigmoid_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_out_tensor_tensor_tensor_double(void* grad_input, void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logit_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_tensor_tensor_double(void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::logit_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_out_tensor_tensor_tensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tanh_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_tensor_tensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::tanh_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_transpose2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_transpose2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_weight)->get(), ((LanternObject<torch::Tensor>*)grad_bias)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<torch::Tensor>*)columns)->get(), ((LanternObject<torch::Tensor>*)ones)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<torch::Tensor>*)columns)->get(), ((LanternObject<torch::Tensor>*)ones)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_transpose3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_transpose3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_weight)->get(), ((LanternObject<torch::Tensor>*)grad_bias)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)output_padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_weight)->get(), ((LanternObject<torch::Tensor>*)grad_bias)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d_forward_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::thnn_conv_depthwise2d_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_weight, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_weight)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_out_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv3d_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_out_tensor_tensor_tensor_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward_out(
        ((LanternObject<torch::Tensor>*)output)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<torch::Tensor>*)bias)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_out_tensor_tensor_tensor_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_weight)->get(), ((LanternObject<torch::Tensor>*)grad_bias)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_tensor_tensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<torch::Tensor>*)finput)->get(), ((LanternObject<torch::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_dilated2d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated2d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_tensor_tensor_intarrayref_tensor_intarrayref_intarrayref_intarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::slow_conv_dilated3d(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<c10::optional<torch::Tensor>>*)optional<torch::Tensor>(bias).get())->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_backward_tensor_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated3d_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)weight)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::col2im_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::col2im(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)output_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::col2im_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::col2im_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* out, void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::im2col_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_tensor_intarrayref_intarrayref_intarrayref_intarrayref(void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::im2col(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_out_tensor_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_input, void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::im2col_backward_out(
        ((LanternObject<torch::Tensor>*)grad_input)->get(), ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_tensor_intarrayref_intarrayref_intarrayref_intarrayref_intarrayref(void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::im2col_backward(
        ((LanternObject<torch::Tensor>*)grad_output)->get(), ((LanternObject<std::vector<int64_t>>*)input_size)->get(), ((LanternObject<std::vector<int64_t>>*)kernel_size)->get(), ((LanternObject<std::vector<int64_t>>*)dilation)->get(), ((LanternObject<std::vector<int64_t>>*)padding)->get(), ((LanternObject<std::vector<int64_t>>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isfinite_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isfinite(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isfinite_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isfinite(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isinf(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isposinf(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isposinf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isposinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isposinf_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isneginf(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isneginf_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().isneginf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_out_tensor_tensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::isneginf_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_batch_dim_tensor_intt_intt(void* self, void* batch_dim, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_add_batch_dim(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)batch_dim)->get(), ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__remove_batch_dim_tensor_intt_intt_intt(void* self, void* level, void* batch_size, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_remove_batch_dim(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)level)->get(), ((LanternObject<int64_t>*)batch_size)->get(), ((LanternObject<int64_t>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_fft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_ifft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_rfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_irfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_hfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_hfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ihfft_tensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_ihfft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_fftn(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_ifftn(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_rfftn(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfftn_tensor_intarrayref_intarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft_irfftn(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)s)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::fft(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fft_tensor_intt_bool(void* self, void* signal_ndim, void* normalized)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().fft(
        ((LanternObject<int64_t>*)signal_ndim)->get(), ((LanternObject<bool>*)normalized)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linalg_det(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::det(
        ((LanternObject<torch::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_det_tensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().det(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::outer(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_outer_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().outer(
        ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_out_tensor_tensor_tensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::outer_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ger(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ger_tensor_tensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(((LanternObject<torch::Tensor>*)self)->get().ger(
        ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_out_tensor_tensor_tensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::ger_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_tensor_scalar_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linalg_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(ord).get())->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_tensor_stdstring_intarrayref_bool_scalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linalg_norm(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_tensor_tensor_scalar_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linalg_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<c10::optional<torch::Scalar>>*)optional<torch::Scalar>(ord).get())->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_tensor_tensor_stdstring_intarrayref_bool_scalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::linalg_norm_out(
        ((LanternObject<torch::Tensor>*)out)->get(), ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<torch::ScalarType>>*)optional<torch::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_serialization_subcmul_tensor_tensor_scalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_test_serialization_subcmul(
        ((LanternObject<torch::Tensor>*)self)->get(), ((LanternObject<torch::Tensor>*)other)->get(), ((LanternObject<torch::Scalar>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_intlist_tensor_intarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_test_optional_intlist(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_filled_intlist_tensor_intarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_test_optional_filled_intlist(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<c10::optional<torch::IntArrayRef>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_floatlist_tensor_arrayrefdouble(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<torch::Tensor>(torch::_test_optional_floatlist(
        ((LanternObject<torch::Tensor>*)values)->get(), ((LanternObject<c10::optional<torch::ArrayRef<double>>>*)addends)->get()));
  LANTERN_FUNCTION_END
}

/* Autogen Body -- End */
