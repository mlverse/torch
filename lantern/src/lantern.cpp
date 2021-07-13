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
void* _lantern__cast_byte_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Byte(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_char_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Char(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_double_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Double(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_float_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Float(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_int_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Int(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_long_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Long(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_short_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Short(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cast_half_attensor_bool(void* self, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cast_Half(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__backward_attensor_attensorlist_attensor_bool_bool(void* self, void* inputs, void* gradient, void* retain_graph, void* create_graph)
{
  LANTERN_FUNCTION_START
    ((LanternObject<at::Tensor>*)self)->get()._backward(((LanternObject<at::TensorList>*)inputs)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(gradient).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(retain_graph).get())->get(), ((LanternObject<bool>*)create_graph)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set_data_attensor_attensor(void* self, void* new_data)
{
  LANTERN_FUNCTION_START
    ((LanternObject<at::Tensor>*)self)->get().set_data(((LanternObject<at::Tensor>*)new_data)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_data_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().data(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_leaf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_leaf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_output_nr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().output_nr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__version_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get()._version(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_requires_grad__attensor_bool(void* self, void* requires_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().requires_grad_(
        ((LanternObject<bool>*)requires_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_retain_grad_attensor(void* self)
{
  LANTERN_FUNCTION_START
    ((LanternObject<at::Tensor>*)self)->get().retain_grad();
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__fw_primal_attensor_intt(void* self, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get()._fw_primal(
        ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__make_dual_attensor_attensor_intt(void* primal, void* tangent, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_make_dual(
        ((LanternObject<at::Tensor>*)primal)->get(), ((LanternObject<at::Tensor>*)tangent)->get(), ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__unpack_dual_attensor_intt(void* dual, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unpack_dual(
        ((LanternObject<at::Tensor>*)dual)->get(), ((LanternObject<int64_t>*)level)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename__attensor_atdimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rename_(
        ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rename_attensor_atdimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rename(
        ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_attensor_atdimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().align_to(
        ((LanternObject<at::DimnameList>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_to_attensor_atdimnamelist_intt(void* self, void* order, void* ellipsis_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().align_to(
        ((LanternObject<at::DimnameList>*)order)->get(), ((LanternObject<int64_t>*)ellipsis_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_align_as_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().align_as(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_align_tensors_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::align_tensors(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__assert_async_attensor(void* self)
{
  LANTERN_FUNCTION_START
    torch::_assert_async(((LanternObject<at::Tensor>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_refine_names_attensor_atdimnamelist(void* self, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().refine_names(
        ((LanternObject<at::DimnameList>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_ctc_loss_attensor_attensor_atintarrayref_atintarrayref_intt(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_ctc_loss(
        ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::IntArrayRef>*)input_lengths)->get(), ((LanternObject<at::IntArrayRef>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_ctc_loss_attensor_attensor_atintarrayref_atintarrayref_intt_bool_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* deterministic, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_ctc_loss(
        ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::IntArrayRef>*)input_lengths)->get(), ((LanternObject<at::IntArrayRef>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__use_cudnn_rnn_flatten_weight()
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_use_cudnn_rnn_flatten_weight(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_flatten_weight_attensorlist_intt_intt_intt_intt_intt_intt_bool_bool(void* weight_arr, void* weight_stride0, void* input_size, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cudnn_rnn_flatten_weight(
        ((LanternObject<at::TensorList>*)weight_arr)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<int64_t>*)input_size)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<bool>*)bidirectional)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_attensor_attensorlist_intt_attensor_attensor_attensor_intt_intt_intt_intt_bool_double_bool_bool_atintarrayref_attensor(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight_buf).get())->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(cx).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<at::IntArrayRef>*)batch_sizes)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(dropout_state).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_rnn_backward_attensor_attensorlist_intt_attensor_attensor_attensor_attensor_attensor_attensor_attensor_intt_intt_intt_intt_bool_double_bool_bool_atintarrayref_attensor_attensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* proj_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_cudnn_rnn_backward(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<at::Tensor>*)weight_buf)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(cx).get())->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_output).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_cy).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)proj_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<at::IntArrayRef>*)batch_sizes)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(dropout_state).get())->get(), ((LanternObject<at::Tensor>*)reserve)->get(), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cudnn_init_dropout_state_double_bool_intt_attensoroptions(void* dropout, void* train, void* dropout_seed, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cudnn_init_dropout_state(
        ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<int64_t>*)dropout_seed)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__debug_has_internal_overlap_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::_debug_has_internal_overlap(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fused_dropout_attensor_double_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fused_dropout(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__masked_scale_attensor_attensor_double(void* self, void* mask, void* scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_masked_scale(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<double>*)scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_draw_attensor_intt_attensor_intt_intt_atscalartype(void* quasi, void* n, void* sobolstate, void* dimension, void* num_generated, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_sobol_engine_draw(
        ((LanternObject<at::Tensor>*)quasi)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<at::Tensor>*)sobolstate)->get(), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_ff__attensor_intt_attensor_intt_intt(void* self, void* n, void* sobolstate, void* dimension, void* num_generated)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sobol_engine_ff_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<at::Tensor>*)sobolstate)->get(), ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)num_generated)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_scramble__attensor_attensor_intt(void* self, void* ltm, void* dimension)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sobol_engine_scramble_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)ltm)->get(), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sobol_engine_initialize_state__attensor_intt(void* self, void* dimension)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sobol_engine_initialize_state_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dimension)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__reshape_from_tensor_attensor_attensor(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_reshape_from_tensor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__shape_as_tensor_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_shape_as_tensor(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout_attensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dropout(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dropout__attensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dropout_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout_attensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::feature_dropout(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_dropout__attensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::feature_dropout_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout_attensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::alpha_dropout(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alpha_dropout__attensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::alpha_dropout_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout_attensor_double_bool(void* input, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::feature_alpha_dropout(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_feature_alpha_dropout__attensor_double_bool(void* self, void* p, void* train)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::feature_alpha_dropout_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<bool>*)train)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::abs(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().abs(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::abs_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_abs__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().abs_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_abs_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::abs_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::absolute(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().absolute(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_absolute__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().absolute_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_absolute_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::absolute_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::angle(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_angle_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().angle(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_angle_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::angle_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_real_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::view_as_real(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_view_as_complex_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::view_as_complex(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sgn(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sgn(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sgn__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sgn_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sgn_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sgn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_real_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::real(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_imag_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::imag(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conj(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_conj_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().conj(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_conj_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conj_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__conj_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_conj(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acos(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().acos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acos_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().acos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acos_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acos_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccos(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arccos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccos_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arccos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccos_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccos_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool1d_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool1d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool1d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_add_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().add(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().add_(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_out_attensor_attensor_attensor_constatscalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::add_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_add_relu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu__attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_add_relu_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_relu_out_attensor_attensor_attensor_constatscalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_add_relu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_add_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().add(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_add__attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().add_(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addmv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat)->get(), ((LanternObject<at::Tensor>*)vec)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addmv(
        ((LanternObject<at::Tensor>*)mat)->get(), ((LanternObject<at::Tensor>*)vec)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addmv_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat)->get(), ((LanternObject<at::Tensor>*)vec)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmv__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addmv_(
        ((LanternObject<at::Tensor>*)mat)->get(), ((LanternObject<at::Tensor>*)vec)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmv_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* mat, void* vec, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addmv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat)->get(), ((LanternObject<at::Tensor>*)vec)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addr(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec1)->get(), ((LanternObject<at::Tensor>*)vec2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addr(
        ((LanternObject<at::Tensor>*)vec1)->get(), ((LanternObject<at::Tensor>*)vec2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addr__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addr_(
        ((LanternObject<at::Tensor>*)vec1)->get(), ((LanternObject<at::Tensor>*)vec2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addr_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* vec1, void* vec2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addr_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec1)->get(), ((LanternObject<at::Tensor>*)vec2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_attensor_atintarrayref_bool(void* theta, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::affine_grid_generator(
        ((LanternObject<at::Tensor>*)theta)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_affine_grid_generator_backward_attensor_atintarrayref_bool(void* grad, void* size, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::affine_grid_generator_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::all(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().all(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_attensor_attensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::all_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::all(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().all(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_all_out_attensor_attensor_atdimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::all_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_allclose_attensor_attensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::allclose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_allclose_attensor_attensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().allclose(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::any(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().any(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_attensor_attensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::any_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::any(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().any(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_any_out_attensor_attensor_atdimname_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::any_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_constatscalar_attensoroptions(void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arange(
        ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_constatscalar_constatscalar_attensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arange(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_constatscalar_constatscalar_constatscalar_attensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arange(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<const at::Scalar &>*)step)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_attensor_constatscalar(void* out, void* end)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arange_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)end)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arange_out_attensor_constatscalar_constatscalar_constatscalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arange_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<const at::Scalar &>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dim_arange_attensor_intt(void* like, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_dim_arange(
        ((LanternObject<at::Tensor>*)like)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmax_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmax_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().argmax(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmax_out_attensor_attensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argmax_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmin_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argmin(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argmin_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().argmin(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argmin_out_attensor_attensor_intt_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argmin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acosh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().acosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acosh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_acosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().acosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_acosh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::acosh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccosh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arccosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccosh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arccosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arccosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arccosh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arccosh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asinh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().asinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asinh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().asinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asinh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asinh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsinh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arcsinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsinh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arcsinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsinh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsinh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atanh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atanh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atanh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atanh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctanh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arctanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctanh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arctanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctanh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctanh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided_attensor_atintarrayref_atintarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::as_strided(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided_attensor_atintarrayref_atintarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().as_strided(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_as_strided__attensor_atintarrayref_atintarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::as_strided_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_as_strided__attensor_atintarrayref_atintarrayref_intt(void* self, void* size, void* stride, void* storage_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().as_strided_(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<c10::optional<int64_t>>*)storage_offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asin(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().asin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asin_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_asin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().asin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_asin_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::asin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsin(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arcsin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsin_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arcsin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arcsin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arcsin_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arcsin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atan(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atan_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_atan_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atan_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctan(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arctan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctan_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_arctan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().arctan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_arctan_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::arctan_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_1d_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atleast_1d(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_1d_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::atleast_1d(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_2d_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atleast_2d(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_2d_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::atleast_2d(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_3d_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atleast_3d(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atleast_3d_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::atleast_3d(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_baddbmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::baddbmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().baddbmm(
        ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_baddbmm__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().baddbmm_(
        ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__baddbmm_mkl__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_baddbmm_mkl_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_baddbmm_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::baddbmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_attensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bartlett_window_intt_bool_attensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bartlett_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_attensor_attensor_attensor_attensor_attensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::batch_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_batch_norm_attensor_attensor_attensor_attensor_attensor_double_double_intt(void* input, void* weight, void* bias, void* mean, void* var, void* eps, void* output_scale, void* output_zero_point)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_batch_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)var)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<double>*)output_scale)->get(), ((LanternObject<int64_t>*)output_zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_attensor_attensor_attensor_attensor_attensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__batch_norm_impl_index_backward_intt_attensor_attensor_attensor_attensor_attensor_attensor_attensor_bool_double_stdarraybool_attensor(void* impl_index, void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var_transform, void* train, void* eps, void* output_mask, void* reservedSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_batch_norm_impl_index_backward(
        ((LanternObject<int64_t>*)impl_index)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_var_transform).get())->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get(), ((LanternObject<at::Tensor>*)reservedSpace)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bernoulli(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bernoulli(
        ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_out_attensor_attensor_atgenerator(void* out, void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bernoulli_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__attensor_attensor_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bernoulli_(
        ((LanternObject<at::Tensor>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli__attensor_double_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bernoulli_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bernoulli_attensor_double_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bernoulli(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bernoulli_attensor_double_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bernoulli(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bilinear_attensor_attensor_attensor_attensor(void* input1, void* input2, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bilinear(
        ((LanternObject<at::Tensor>*)input1)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_attensor_attensor_attensor_intt(void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_out_attensor_attensor_attensor_attensor_intt(void* out, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_attensor_attensor_attensor_attensor_intt(void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_backward_out_attensor_attensor_attensor_attensor_attensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_attensor_attensor_attensor_attensor_intt(void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy_with_logits(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(pos_weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binary_cross_entropy_with_logits_backward_attensor_attensor_attensor_attensor_attensor_intt(void* grad_output, void* self, void* target, void* weight, void* pos_weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binary_cross_entropy_with_logits_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(pos_weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bincount_attensor_attensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bincount(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weights).get())->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bincount_attensor_attensor_intt(void* self, void* weights, void* minlength)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bincount(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weights).get())->get(), ((LanternObject<int64_t>*)minlength)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_not(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_not__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_not_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_not_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::copysign_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::copysign(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().copysign(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().copysign_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::copysign(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().copysign(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copysign__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().copysign_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copysign_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::copysign_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_not(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_not(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_not__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_not_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_not_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_not_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_xor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_xor(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_xor__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_xor_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_xor_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_xor_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_and(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_and(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_and__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_and_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_and_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_and_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_or(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_or(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logical_or__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logical_or_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logical_or_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logical_or_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_attensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_blackman_window_intt_bool_attensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::blackman_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bmm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bmm(
        ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_attensor_attensor_bool(void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_bmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bmm_out_attensor_attensor_attensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__bmm_out_attensor_attensor_attensor_bool(void* out, void* self, void* mat2, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_bmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_broadcast_tensors_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::broadcast_tensors(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_broadcast_to_attensor_atintarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::broadcast_to(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_broadcast_to_attensor_atintarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().broadcast_to(
        ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_attensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cat(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_attensor_attensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cat_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_attensorlist_atdimname(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cat(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cat_out_attensor_attensorlist_atdimname(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cat_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_block_diag_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::block_diag(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ceil(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ceil(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ceil_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ceil__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ceil_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_ceil_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ceil_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chain_matmul_attensorlist(void* matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::chain_matmul(
        ((LanternObject<at::TensorList>*)matrices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chain_matmul_out_attensor_attensorlist(void* out, void* matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::chain_matmul_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)matrices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_chunk_attensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unsafe_chunk(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_chunk_attensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().unsafe_chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_chunk_attensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::chunk(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_chunk_attensor_intt_intt(void* self, void* chunks, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().chunk(
        ((LanternObject<int64_t>*)chunks)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_attensor_intt_intt(void* self, void* sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::tensor_split(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_attensor_intt_intt(void* self, void* sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().tensor_split(
        ((LanternObject<int64_t>*)sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_attensor_atintarrayref_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::tensor_split(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_attensor_atintarrayref_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().tensor_split(
        ((LanternObject<at::IntArrayRef>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensor_split_attensor_attensor_intt(void* self, void* tensor_indices_or_sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::tensor_split(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor_indices_or_sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tensor_split_attensor_attensor_intt(void* self, void* tensor_indices_or_sections, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().tensor_split(
        ((LanternObject<at::Tensor>*)tensor_indices_or_sections)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp(
        ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp__attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp__attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_(
        ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp__attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp__attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_out_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_out_attensor_attensor_attensor_attensor(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_attensor_constatscalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max_attensor_constatscalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_max(
        ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_attensor_attensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max_attensor_attensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_max(
        ((LanternObject<at::Tensor>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max__attensor_constatscalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max__attensor_constatscalar(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_max_(
        ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max__attensor_attensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_max__attensor_attensor(void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_max_(
        ((LanternObject<at::Tensor>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_out_attensor_attensor_constatscalar(void* out, void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_max_out_attensor_attensor_attensor(void* out, void* self, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_max_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_attensor_constatscalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min_attensor_constatscalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_min(
        ((LanternObject<const at::Scalar &>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_attensor_attensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min_attensor_attensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_min(
        ((LanternObject<at::Tensor>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min__attensor_constatscalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min__attensor_constatscalar(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_min_(
        ((LanternObject<const at::Scalar &>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min__attensor_attensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clamp_min__attensor_attensor(void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clamp_min_(
        ((LanternObject<at::Tensor>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_out_attensor_attensor_constatscalar(void* out, void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clamp_min_out_attensor_attensor_attensor(void* out, void* self, void* min)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clamp_min_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)min)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip_attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clip(
        ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip_attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clip(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip__attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip__attensor_constatscalar_constatscalar(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clip_(
        ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip__attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clip__attensor_attensor_attensor(void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clip_(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_out_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)min)->get(), ((LanternObject<c10::optional<at::Scalar>>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clip_out_attensor_attensor_attensor_attensor(void* out, void* self, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clip_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(min).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(max).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_is_acceptable_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::cudnn_is_acceptable(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_attensor_attensor(void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::complex(
        ((LanternObject<at::Tensor>*)real)->get(), ((LanternObject<at::Tensor>*)imag)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_complex_out_attensor_attensor_attensor(void* out, void* real, void* imag)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::complex_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)real)->get(), ((LanternObject<at::Tensor>*)imag)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_attensor_attensor(void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::polar(
        ((LanternObject<at::Tensor>*)abs)->get(), ((LanternObject<at::Tensor>*)angle)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_polar_out_attensor_attensor_attensor(void* out, void* abs, void* angle)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::polar_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)abs)->get(), ((LanternObject<at::Tensor>*)angle)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_constant_pad_nd_attensor_atintarrayref_constatscalar(void* self, void* pad, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::constant_pad_nd(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)pad)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_contiguous_attensor_atmemoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().contiguous(
        ((LanternObject<at::MemoryFormat>*)memory_format)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::convolution(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_overrideable_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::convolution_overrideable(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_convolution_backward_overrideable_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt_stdarraybool(void* grad_output, void* input, void* weight, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::convolution_backward_overrideable(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt_bool_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_convolution(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt_bool_bool_bool(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_convolution(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_mode_attensor_attensor_attensor_atintarrayref_stdstring_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_convolution_mode(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_nogroup_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* transposed, void* output_padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_convolution_nogroup(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__convolution_double_backward_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_atintarrayref_intt_bool_bool_bool_bool_stdarraybool(void* ggI, void* ggW, void* ggb, void* gO, void* weight, void* self, void* stride, void* padding, void* dilation, void* transposed, void* output_padding, void* groups, void* benchmark, void* deterministic, void* cudnn_enabled, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_convolution_double_backward(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(ggI).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(ggW).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(ggb).get())->get(), ((LanternObject<at::Tensor>*)gO)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)transposed)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)cudnn_enabled)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv1d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv1d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv2d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv3d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv1d_attensor_attensor_attensor_atintarrayref_stdstring_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv1d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv2d_attensor_attensor_attensor_atintarrayref_stdstring_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv3d_attensor_attensor_attensor_atintarrayref_stdstring_atintarrayref_intt(void* input, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<std::string>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_attensor_attensor_attensor_intt(void* self, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv_tbc(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)bias)->get(), ((LanternObject<int64_t>*)pad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_tbc_backward_attensor_attensor_attensor_attensor_intt(void* self, void* input, void* weight, void* bias, void* pad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_tbc_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)bias)->get(), ((LanternObject<int64_t>*)pad)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose1d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_atintarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv_transpose1d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose2d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_atintarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv_transpose2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_transpose3d_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_atintarrayref(void* input, void* weight, void* bias, void* stride, void* padding, void* output_padding, void* groups, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv_transpose3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_copy__attensor_attensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().copy_(
        ((LanternObject<at::Tensor>*)src)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__copy_from_attensor_attensor_bool(void* self, void* dst, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_copy_from(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)dst)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cos(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cos(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cos_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cos__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cos_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cos_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cos_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cosh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cosh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cosh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cosh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cosh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_cosh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cosh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_embedding_loss_attensor_attensor_attensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cosine_embedding_loss(
        ((LanternObject<at::Tensor>*)input1)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_attensor_atintarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::count_nonzero(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_attensor_atintarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().count_nonzero(
        ((LanternObject<at::IntArrayRef>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_count_nonzero_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::count_nonzero(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_count_nonzero_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().count_nonzero(
        ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_attensor_intt_intt_intt_intt(void* theta, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_affine_grid_generator(
        ((LanternObject<at::Tensor>*)theta)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_affine_grid_generator_backward_attensor_intt_intt_intt_intt(void* grad, void* N, void* C, void* H, void* W)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_affine_grid_generator_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)H)->get(), ((LanternObject<int64_t>*)W)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_attensor_attensor_attensor_attensor_attensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_batch_norm_backward_attensor_attensor_attensor_attensor_attensor_attensor_attensor_double_attensor(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon, void* reserveSpace)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_batch_norm_backward(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_var).get())->get(), ((LanternObject<double>*)epsilon)->get(), ((LanternObject<at::Tensor>*)reserveSpace)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_input_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_backward_input(
        ((LanternObject<at::IntArrayRef>*)self_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_backward_weight_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_backward_weight(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* self, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_convolution_transpose_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_input_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_transpose_backward_input(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_transpose_backward_weight_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* allow_tf32)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_transpose_backward_weight(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<bool>*)allow_tf32)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_relu_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* self, void* weight, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_relu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_convolution_add_relu_attensor_attensor_attensor_constatscalar_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* self, void* weight, void* z, void* alpha, void* bias, void* stride, void* padding, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_convolution_add_relu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)z)->get(), ((LanternObject<c10::optional<at::Scalar>>*)alpha)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_attensor_attensor(void* self, void* grid)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cudnn_grid_sampler(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grid)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cudnn_grid_sampler_backward_attensor_attensor_attensor(void* self, void* grid, void* grad_output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cudnn_grid_sampler_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<at::Tensor>*)grad_output)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().cummax(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_attensor_attensor_attensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummax_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().cummax(
        ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummax_out_attensor_attensor_attensor_atdimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummax_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummax_helper_attensor_attensor_attensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummax_helper(((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().cummin(
        ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_attensor_attensor_attensor_intt(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cummin_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().cummin(
        ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cummin_out_attensor_attensor_attensor_atdimname(void* values, void* indices, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::cummin_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__cummin_helper_attensor_attensor_attensor_intt(void* self, void* values, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    torch::_cummin_helper(((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_cummaxmin_backward_attensor_attensor_attensor_intt(void* grad, void* input, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cummaxmin_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumprod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumprod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod__attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumprod_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_attensor_attensor_intt_atscalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumprod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumprod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumprod(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumprod__attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumprod_(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_out_attensor_attensor_atdimname_atscalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumprod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumprod_backward_attensor_attensor_intt_attensor(void* grad, void* input, void* dim, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumprod_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumsum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumsum(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum__attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumsum_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_attensor_attensor_intt_atscalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumsum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumsum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumsum(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cumsum__attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cumsum_(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cumsum_out_attensor_attensor_atdimname_atscalartype(void* out, void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cumsum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_attensor_attensor_atintarrayref_atintarrayref_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ctc_loss(
        ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::IntArrayRef>*)input_lengths)->get(), ((LanternObject<at::IntArrayRef>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ctc_loss_attensor_attensor_attensor_attensor_intt_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* reduction, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ctc_loss(
        ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::Tensor>*)input_lengths)->get(), ((LanternObject<at::Tensor>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_attensor_attensor_atintarrayref_atintarrayref_intt_bool(void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_ctc_loss(
        ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::IntArrayRef>*)input_lengths)->get(), ((LanternObject<at::IntArrayRef>*)target_lengths)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__ctc_loss_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_attensor_attensor_intt_bool(void* grad, void* log_probs, void* targets, void* input_lengths, void* target_lengths, void* neg_log_likelihood, void* log_alpha, void* blank, void* zero_infinity)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_ctc_loss_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)log_probs)->get(), ((LanternObject<at::Tensor>*)targets)->get(), ((LanternObject<at::IntArrayRef>*)input_lengths)->get(), ((LanternObject<at::IntArrayRef>*)target_lengths)->get(), ((LanternObject<at::Tensor>*)neg_log_likelihood)->get(), ((LanternObject<at::Tensor>*)log_alpha)->get(), ((LanternObject<int64_t>*)blank)->get(), ((LanternObject<bool>*)zero_infinity)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_embed_attensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diag_embed(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_embed_attensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diag_embed(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagflat_attensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diagflat(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagflat_attensor_intt(void* self, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diagflat(
        ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_attensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diagonal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_attensor_intt_intt_intt(void* self, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diagonal(
        ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_attensor_atdimname_atdimname_atdimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diagonal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)outdim)->get(), ((LanternObject<at::Dimname>*)dim1)->get(), ((LanternObject<at::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diagonal_attensor_atdimname_atdimname_atdimname_intt(void* self, void* outdim, void* dim1, void* dim2, void* offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diagonal(
        ((LanternObject<at::Dimname>*)outdim)->get(), ((LanternObject<at::Dimname>*)dim1)->get(), ((LanternObject<at::Dimname>*)dim2)->get(), ((LanternObject<int64_t>*)offset)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diagonal_backward_attensor_atintarrayref_intt_intt_intt(void* grad, void* input_sizes, void* offset, void* dim1, void* dim2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diagonal_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)input_sizes)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<int64_t>*)dim1)->get(), ((LanternObject<int64_t>*)dim2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill_diagonal__attensor_constatscalar_bool(void* self, void* fill_value, void* wrap)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fill_diagonal_(
        ((LanternObject<const at::Scalar &>*)fill_value)->get(), ((LanternObject<bool>*)wrap)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diff_attensor_intt_intt_attensor_attensor(void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diff(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(prepend).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(append).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diff_attensor_intt_intt_attensor_attensor(void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diff(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(prepend).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(append).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diff_out_attensor_attensor_intt_intt_attensor_attensor(void* out, void* self, void* n, void* dim, void* prepend, void* append)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diff_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(prepend).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(append).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_constatscalar_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)spacing)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_constatscalar_atintarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)spacing)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_atintarrayref_intt(void* self, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_atarrayrefatscalar_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)spacing)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_atarrayrefatscalar_atintarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)spacing)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_attensorlist_intt_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorList>*)spacing)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gradient_attensor_attensorlist_atintarrayref_intt(void* self, void* spacing, void* dim, void* edge_order)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::gradient(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorList>*)spacing)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)edge_order)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div_(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_out_attensor_attensor_attensor_stdstring(void* out, void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_div_attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::div(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div_attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_div__attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().div_(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__attensor_attensor_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide_(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_out_attensor_attensor_attensor_stdstring(void* out, void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_divide_attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide_attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_divide__attensor_constatscalar_stdstring(void* self, void* other, void* rounding_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().divide_(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(rounding_mode).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::true_divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().true_divide(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().true_divide_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::true_divide_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_true_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::true_divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().true_divide(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_true_divide__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().true_divide_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dot_attensor_attensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dot(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dot_attensor_attensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().dot(
        ((LanternObject<at::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dot_out_attensor_attensor_attensor(void* out, void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dot_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::vdot(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vdot_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().vdot(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vdot_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::vdot_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_einsum_stdstring_attensorlist(void* equation, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::einsum(
        ((LanternObject<std::string>*)equation)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_attensor_attensor_intt_bool_bool(void* weight, void* indices, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::embedding(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_backward_attensor_attensor_intt_intt_bool_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq, void* sparse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::embedding_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<bool>*)sparse)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_dense_backward_attensor_attensor_intt_intt_bool(void* grad_output, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::embedding_dense_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_renorm__attensor_attensor_double_double(void* self, void* indices, void* max_norm, void* norm_type)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::embedding_renorm_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<double>*)max_norm)->get(), ((LanternObject<double>*)norm_type)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_sparse_backward_attensor_attensor_intt_intt_bool(void* grad, void* indices, void* num_weights, void* padding_idx, void* scale_grad_by_freq)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::embedding_sparse_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<int64_t>*)padding_idx)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_forward_only_attensor_attensor_attensor_bool_intt_bool_attensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag_forward_only(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<int64_t>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__rowwise_prune_attensor_attensor_atscalartype(void* weight, void* mask, void* compressed_indices_dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_rowwise_prune(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::ScalarType>*)compressed_indices_dtype)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_row_stack_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::row_stack(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_row_stack_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::row_stack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_bag_attensor_attensor_attensor_bool_intt_bool_attensor_bool(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::embedding_bag(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_embedding_bag_attensor_attensor_attensor_bool_intt_bool_attensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::embedding_bag(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<c10::optional<int64_t>>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_attensor_attensor_attensor_bool_intt_bool_attensor_bool_intt(void* weight, void* indices, void* offsets, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* include_last_offset, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_embedding_bag(
        ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<bool>*)include_last_offset)->get(), ((LanternObject<int64_t>*)padding_idx)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_backward_attensor_attensor_attensor_attensor_attensor_attensor_intt_bool_intt_bool_attensor_intt(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* sparse, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_embedding_bag_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<at::Tensor>*)offset2bag)->get(), ((LanternObject<at::Tensor>*)bag_size)->get(), ((LanternObject<at::Tensor>*)maximum_indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<bool>*)sparse)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_sparse_backward_attensor_attensor_attensor_attensor_attensor_intt_bool_intt_attensor_intt(void* grad, void* indices, void* offsets, void* offset2bag, void* bag_size, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_embedding_bag_sparse_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<at::Tensor>*)offset2bag)->get(), ((LanternObject<at::Tensor>*)bag_size)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_dense_backward_attensor_attensor_attensor_attensor_attensor_intt_bool_intt_attensor_intt(void* grad, void* indices, void* offset2bag, void* bag_size, void* maximum_indices, void* num_weights, void* scale_grad_by_freq, void* mode, void* per_sample_weights, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_embedding_bag_dense_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offset2bag)->get(), ((LanternObject<at::Tensor>*)bag_size)->get(), ((LanternObject<at::Tensor>*)maximum_indices)->get(), ((LanternObject<int64_t>*)num_weights)->get(), ((LanternObject<bool>*)scale_grad_by_freq)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(per_sample_weights).get())->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__embedding_bag_per_sample_weights_backward_attensor_attensor_attensor_attensor_attensor_intt_intt(void* grad, void* weight, void* indices, void* offsets, void* offset2bag, void* mode, void* padding_idx)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_embedding_bag_per_sample_weights_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)offsets)->get(), ((LanternObject<at::Tensor>*)offset2bag)->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)padding_idx)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_atintarrayref_atdimnamelist_attensoroptions_atmemoryformat(void* size, void* names, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_atintarrayref_attensoroptions_atmemoryformat(void* size, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_empty_attensor_atintarrayref_attensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().new_empty(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_empty_strided_attensor_atintarrayref_atintarrayref_attensoroptions(void* self, void* size, void* stride, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().new_empty_strided(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_full_attensor_atintarrayref_constatscalar_attensoroptions(void* self, void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().new_full(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<const at::Scalar &>*)fill_value)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_new_zeros_attensor_atintarrayref_attensoroptions(void* self, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().new_zeros(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_affine_quantized_atintarrayref_attensoroptions_double_intt_atmemoryformat(void* size, void* options, void* scale, void* zero_point, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_empty_affine_quantized(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__empty_per_channel_affine_quantized_atintarrayref_attensor_attensor_intt_attensoroptions_atmemoryformat(void* size, void* scales, void* zero_points, void* axis, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_empty_per_channel_affine_quantized(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::Tensor>*)scales)->get(), ((LanternObject<at::Tensor>*)zero_points)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize__attensor_atintarrayref_atmemoryformat(void* self, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().resize_(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_quantized_atintarrayref_attensor(void* size, void* qtensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty_quantized(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::Tensor>*)qtensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_out_attensor_atintarrayref_atmemoryformat(void* out, void* size, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_like_attensor_attensoroptions_atmemoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_empty_strided_atintarrayref_atintarrayref_attensoroptions(void* size, void* stride, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::empty_strided(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erf(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erf_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erf__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erf_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erf_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erf_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erfc(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erfc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erfc_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erfc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfc_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erfc_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().exp_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp2(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().exp2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp2_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exp2__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().exp2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_exp2_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::exp2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::expm1(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().expm1(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::expm1_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expm1__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().expm1_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_expm1_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::expm1_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_attensor_atintarrayref_bool(void* self, void* size, void* implicit)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().expand(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<bool>*)implicit)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_expand_as_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().expand_as(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_attensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_intt_intt_attensoroptions(void* n, void* m, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eye(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_attensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eye_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eye_out_attensor_intt_intt(void* out, void* n, void* m)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eye_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<int64_t>*)m)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_attensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flatten(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_attensor_intt_intt(void* self, void* start_dim, void* end_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_attensor_intt_intt_atdimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flatten(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_attensor_intt_intt_atdimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flatten(
        ((LanternObject<int64_t>*)start_dim)->get(), ((LanternObject<int64_t>*)end_dim)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_attensor_atdimname_atdimname_atdimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flatten(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)start_dim)->get(), ((LanternObject<at::Dimname>*)end_dim)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_attensor_atdimname_atdimname_atdimname(void* self, void* start_dim, void* end_dim, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flatten(
        ((LanternObject<at::Dimname>*)start_dim)->get(), ((LanternObject<at::Dimname>*)end_dim)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_attensor_atdimnamelist_atdimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flatten(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dims)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flatten_attensor_atdimnamelist_atdimname(void* self, void* dims, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flatten(
        ((LanternObject<at::DimnameList>*)dims)->get(), ((LanternObject<at::Dimname>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_attensor_intt_atintarrayref_atdimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().unflatten(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::IntArrayRef>*)sizes)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unflatten_attensor_atdimname_atintarrayref_atdimnamelist(void* self, void* dim, void* sizes, void* names)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().unflatten(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::IntArrayRef>*)sizes)->get(), ((LanternObject<at::DimnameList>*)names)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__attensor_constatscalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fill_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__attensor_constatscalar(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fill_(
        ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fill__attensor_attensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fill_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fill__attensor_attensor(void* self, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fill_(
        ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor_divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor_divide(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor_divide_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor_divide_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_floor_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::floor_divide(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor_divide(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_floor_divide__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().floor_divide_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frac(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().frac(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frac_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frac__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().frac_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_frac_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frac_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_atintarrayref_constatscalar_atdimnamelist_attensoroptions(void* size, void* fill_value, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::full(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<const at::Scalar &>*)fill_value)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_atintarrayref_constatscalar_attensoroptions(void* size, void* fill_value, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::full(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<const at::Scalar &>*)fill_value)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_out_attensor_atintarrayref_constatscalar(void* out, void* size, void* fill_value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::full_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<const at::Scalar &>*)fill_value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_full_like_attensor_constatscalar_attensoroptions_atmemoryformat(void* self, void* fill_value, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::full_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)fill_value)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_from_file_stdstring_bool_intt_attensoroptions(void* filename, void* shared, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::from_file(
        ((LanternObject<std::string>*)filename)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(shared).get())->get(), ((LanternObject<c10::optional<int64_t>>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gcd_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gcd(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gcd(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gcd__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gcd_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gcd__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gcd_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lcm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lcm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lcm(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lcm__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lcm_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lcm__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lcm_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_attensor_attensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::grid_sampler(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_attensor_attensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::grid_sampler_2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_2d_backward_attensor_attensor_attensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_attensor_attensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_grid_sampler_2d_cpu_fallback(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__grid_sampler_2d_cpu_fallback_backward_attensor_attensor_attensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_grid_sampler_2d_cpu_fallback_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_attensor_attensor_intt_intt_bool(void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::grid_sampler_3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_grid_sampler_3d_backward_attensor_attensor_attensor_intt_intt_bool(void* grad_output, void* input, void* grid, void* interpolation_mode, void* padding_mode, void* align_corners)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::grid_sampler_3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grid)->get(), ((LanternObject<int64_t>*)interpolation_mode)->get(), ((LanternObject<int64_t>*)padding_mode)->get(), ((LanternObject<bool>*)align_corners)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_attensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hann_window_intt_bool_attensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hann_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_attensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_attensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_attensoroptions(void* window_length, void* periodic, void* alpha, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hamming_window_intt_bool_double_double_attensoroptions(void* window_length, void* periodic, void* alpha, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hamming_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)alpha)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_attensoroptions(void* window_length, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_attensoroptions(void* window_length, void* periodic, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kaiser_window_intt_bool_double_attensoroptions(void* window_length, void* periodic, void* beta, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kaiser_window(
        ((LanternObject<int64_t>*)window_length)->get(), ((LanternObject<bool>*)periodic)->get(), ((LanternObject<double>*)beta)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hinge_embedding_loss_attensor_attensor_double_intt(void* self, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hinge_embedding_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_group_norm_attensor_intt_attensor_attensor_double_bool(void* input, void* num_groups, void* weight, void* bias, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::group_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<int64_t>*)num_groups)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_attensor_attensor_attensor_intt_intt_intt_intt_double(void* input, void* weight, void* bias, void* N, void* C, void* HxW, void* group, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_group_norm_backward_attensor_attensor_attensor_attensor_attensor_intt_intt_intt_intt_stdarraybool(void* grad_out, void* input, void* mean, void* rstd, void* weight, void* N, void* C, void* HxW, void* group, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_group_norm_backward(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)rstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)N)->get(), ((LanternObject<int64_t>*)C)->get(), ((LanternObject<int64_t>*)HxW)->get(), ((LanternObject<int64_t>*)group)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_r2c_attensor_atintarrayref_intt_bool(void* self, void* dim, void* normalization, void* onesided)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_r2c(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_r2c_out_attensor_attensor_atintarrayref_intt_bool(void* out, void* self, void* dim, void* normalization, void* onesided)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_r2c_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)onesided)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2r_attensor_atintarrayref_intt_intt(void* self, void* dim, void* normalization, void* last_dim_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_c2r(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<int64_t>*)last_dim_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2r_out_attensor_attensor_atintarrayref_intt_intt(void* out, void* self, void* dim, void* normalization, void* last_dim_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_c2r_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<int64_t>*)last_dim_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2c_attensor_atintarrayref_intt_bool(void* self, void* dim, void* normalization, void* forward)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_c2c(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)forward)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fft_c2c_out_attensor_attensor_atintarrayref_intt_bool(void* out, void* self, void* dim, void* normalization, void* forward)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fft_c2c_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<int64_t>*)normalization)->get(), ((LanternObject<bool>*)forward)->get()));
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

void* _lantern_index_attensor_constclistcoptionalattensor(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_attensor_constclistcoptionalattensor(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index(
        ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_copy_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_copy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_copy(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy__attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_copy_(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_copy_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_copy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_copy_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_copy(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put__attensor_constclistcoptionalattensor_attensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_put_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put__attensor_constclistcoptionalattensor_attensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_put_(
        ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_put_attensor_constclistcoptionalattensor_attensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_put(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_put_attensor_constclistcoptionalattensor_attensor_bool(void* self, void* indices, void* values, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_put(
        ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__index_put_impl__attensor_constclistcoptionalattensor_attensor_bool_bool(void* self, void* indices, void* values, void* accumulate, void* unsafe)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_index_put_impl_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const c10::List<c10::optional<at::Tensor>> &>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<bool>*)accumulate)->get(), ((LanternObject<bool>*)unsafe)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_instance_norm_attensor_attensor_attensor_attensor_attensor_bool_double_double_bool(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* use_input_stats, void* momentum, void* eps, void* cudnn_enabled)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::instance_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)use_input_stats)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enabled)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::inverse(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_inverse_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().inverse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_inverse_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::inverse_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__inverse_helper_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_inverse_helper(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isclose_attensor_attensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isclose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isclose_attensor_attensor_double_double_bool(void* self, void* other, void* rtol, void* atol, void* equal_nan)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isclose(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<double>*)rtol)->get(), ((LanternObject<double>*)atol)->get(), ((LanternObject<bool>*)equal_nan)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isnan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isnan(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isnan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isnan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_distributed_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_distributed(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_distributed_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_distributed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_floating_point_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_floating_point(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_floating_point_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_floating_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_complex_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_complex(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_complex_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_complex(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isreal_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isreal(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isreal_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isreal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_nonzero_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_nonzero(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_nonzero_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_is_same_size_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_same_size(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_same_size_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_same_size(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_is_signed_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::is_signed(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_signed_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_signed(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_attensor_attensor_intt_bool(void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kl_div(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kl_div_backward_attensor_attensor_attensor_intt_bool(void* grad_output, void* self, void* target, void* reduction, void* log_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kl_div_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<bool>*)log_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kron_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kron(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kron_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().kron(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kron_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::kron_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_attensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_attensor_intt_intt_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_attensor_attensor_attensor_intt_intt_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_attensor_intt_atdimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_kthvalue_attensor_intt_atdimname_bool(void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().kthvalue(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_kthvalue_out_attensor_attensor_attensor_intt_atdimname_bool(void* values, void* indices, void* self, void* k, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::kthvalue_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_layer_norm_attensor_atintarrayref_attensor_attensor_double_bool(void* input, void* normalized_shape, void* weight, void* bias, void* eps, void* cudnn_enable)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::layer_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)normalized_shape)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)cudnn_enable)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_attensor_atintarrayref_attensor_attensor_double(void* input, void* normalized_shape, void* weight, void* bias, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)normalized_shape)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_layer_norm_backward_attensor_attensor_atintarrayref_attensor_attensor_attensor_attensor_stdarraybool(void* grad_out, void* input, void* normalized_shape, void* mean, void* rstd, void* weight, void* bias, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_layer_norm_backward(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)normalized_shape)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)rstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num_attensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nan_to_num(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nan_to_num_attensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nan_to_num(
        ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num__attensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nan_to_num_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nan_to_num__attensor_double_double_double(void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nan_to_num_(
        ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nan_to_num_out_attensor_attensor_double_double_double(void* out, void* self, void* nan, void* posinf, void* neginf)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nan_to_num_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)nan)->get(), ((LanternObject<c10::optional<double>>*)posinf)->get(), ((LanternObject<c10::optional<double>>*)neginf)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linear_attensor_attensor_attensor(void* input, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linear(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_attensor_attensor_attensor(void* self, void* weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_linear(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_input_atintarrayref_attensor_attensor(void* input_size, void* grad_output, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_linear_backward_input(
        ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_weights_attensor_attensor_attensor_bool(void* grad_output, void* input, void* weight, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_linear_backward_weights(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<bool>*)bias_defined)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_linear_backward_attensor_attensor_attensor_stdarraybool(void* self, void* grad_output, void* weight, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_linear_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_fp32_activation_attensor_attensor_attensor_attensor_constatscalar_constatscalar_attensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_linear_int8_weight_fp32_activation(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)packed)->get(), ((LanternObject<at::Tensor>*)col_offsets)->get(), ((LanternObject<const at::Scalar &>*)weight_scale)->get(), ((LanternObject<const at::Scalar &>*)weight_zero_point)->get(), ((LanternObject<at::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_int8_weight_attensor_attensor_attensor_attensor_constatscalar_constatscalar_attensor(void* input, void* weight, void* packed, void* col_offsets, void* weight_scale, void* weight_zero_point, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_linear_int8_weight(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::Tensor>*)packed)->get(), ((LanternObject<at::Tensor>*)col_offsets)->get(), ((LanternObject<const at::Scalar &>*)weight_scale)->get(), ((LanternObject<const at::Scalar &>*)weight_zero_point)->get(), ((LanternObject<at::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_quantize_weight_attensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fbgemm_linear_quantize_weight(
        ((LanternObject<at::Tensor>*)input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_gemm_matrix_fp16_attensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_pack_gemm_matrix_fp16(
        ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_fp32_activation_attensor_attensor_attensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_linear_fp16_weight_fp32_activation(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)packed_weight)->get(), ((LanternObject<at::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_linear_fp16_weight_attensor_attensor_attensor(void* input, void* packed_weight, void* bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_linear_fp16_weight(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)packed_weight)->get(), ((LanternObject<at::Tensor>*)bias)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_attensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fbgemm_pack_quantized_matrix_attensor_intt_intt(void* input, void* K, void* N)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fbgemm_pack_quantized_matrix(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<int64_t>*)K)->get(), ((LanternObject<int64_t>*)N)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ldexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ldexp_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ldexp(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ldexp_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ldexp__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ldexp_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ldexp_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ldexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_constatscalar_constatscalar_intt_attensoroptions(void* start, void* end, void* steps, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linspace(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linspace_out_attensor_constatscalar_constatscalar_intt(void* out, void* start, void* end, void* steps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linspace_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log10(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log10(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log10_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log10__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log10_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log10_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log10_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log1p(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log1p(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log1p_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log1p__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log1p_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log1p_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log1p_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log2(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log2(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log2_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log2__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log2_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_log2_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logaddexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logaddexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logaddexp(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logaddexp2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logaddexp2_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logaddexp2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logaddexp2_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logaddexp2(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().xlogy(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_constatscalar_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy(
        ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().xlogy(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().xlogy_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_xlogy__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().xlogy_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_attensor_constatscalar_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_xlogy_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::xlogy_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logdet_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logdet(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logdet_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logdet(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_constatscalar_constatscalar_intt_double_attensoroptions(void* start, void* end, void* steps, void* base, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logspace(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logspace_out_attensor_constatscalar_constatscalar_intt_double(void* out, void* start, void* end, void* steps, void* base)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logspace_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<c10::optional<int64_t>>*)steps)->get(), ((LanternObject<double>*)base)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log_softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log_softmax(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_attensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__log_softmax_backward_data_attensor_attensor_intt_attensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_log_softmax_backward_data(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_logcumsumexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__logcumsumexp_out_attensor_attensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_logcumsumexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logcumsumexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logcumsumexp(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_attensor_attensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logcumsumexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logcumsumexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logcumsumexp_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logcumsumexp(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logcumsumexp_out_attensor_attensor_atdimname(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logcumsumexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logsumexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logsumexp(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_attensor_attensor_atintarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logsumexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_attensor_atdimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logsumexp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logsumexp_attensor_atdimnamelist_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logsumexp(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logsumexp_out_attensor_attensor_atdimnamelist_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logsumexp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_margin_ranking_loss_attensor_attensor_attensor_double_intt(void* input1, void* input2, void* target, void* margin, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::margin_ranking_loss(
        ((LanternObject<at::Tensor>*)input1)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matmul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matmul_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().matmul(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matmul_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matmul_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_attensor_double_bool(void* self, void* tol, void* symmetric)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_rank(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)tol)->get(), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_rank_attensor_bool(void* self, void* symmetric)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_rank(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)symmetric)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_power_attensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_power(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_power_attensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().matrix_power(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_power_out_attensor_attensor_intt(void* out, void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_power_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_exp(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_matrix_exp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().matrix_exp(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_matrix_exp_backward_attensor_attensor(void* self, void* grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::matrix_exp_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__aminmax_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_aminmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_attensor_attensor(void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_compute_linear_combination(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)coefficients)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__compute_linear_combination_out_attensor_attensor_attensor(void* out, void* input, void* coefficients)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_compute_linear_combination_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)coefficients)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().max(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_attensor_attensor_attensor_intt_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        ((LanternObject<at::Tensor>*)max)->get(), ((LanternObject<at::Tensor>*)max_values)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().max(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_attensor_attensor_attensor_atdimname_bool(void* max, void* max_values, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_out(
        ((LanternObject<at::Tensor>*)max)->get(), ((LanternObject<at::Tensor>*)max_values)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_value_selecting_reduction_backward_attensor_intt_attensor_atintarrayref_bool(void* grad, void* dim, void* indices, void* sizes, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::value_selecting_reduction_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)sizes)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::amax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amax_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().amax(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amax_out_attensor_attensor_atintarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::amax_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_with_indices_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool1d_with_indices(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool1d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool2d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_max_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool2d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* grad_output, void* output, void* input, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_max_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool3d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_max_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_max_pool3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* grad_output, void* output, void* input, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_max_pool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool1d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_max_pool1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_max_pool2d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_max_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mean(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mean(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_attensor_attensor_atintarrayref_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mean_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_attensor_atdimnamelist_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mean_attensor_atdimnamelist_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mean(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mean_out_attensor_attensor_atdimnamelist_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mean_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_median_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::median(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().median(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_median_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().median(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_attensor_attensor_attensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_median_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().median(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_median_out_attensor_attensor_attensor_atdimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::median_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanmedian(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nanmedian(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().nanmedian(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_out_attensor_attensor_attensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanmedian_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().nanmedian(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nanmedian_out_attensor_attensor_attensor_atdimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nanmedian_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().min(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_attensor_attensor_attensor_intt_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        ((LanternObject<at::Tensor>*)min)->get(), ((LanternObject<at::Tensor>*)min_indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().min(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_attensor_attensor_attensor_atdimname_bool(void* min, void* min_indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::min_out(
        ((LanternObject<at::Tensor>*)min)->get(), ((LanternObject<at::Tensor>*)min_indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::amin(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_amin_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().amin(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_amin_out_attensor_attensor_atintarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::amin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_input_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_convolution_backward_input(
        ((LanternObject<at::IntArrayRef>*)self_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_weights_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* bias_defined)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward_weights(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)bias_defined)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_convolution_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mkldnn_convolution_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_attensor_attensor_attensor_attensor_attensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* exponential_average_factor, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)exponential_average_factor)->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_batch_norm_backward_attensor_attensor_attensor_attensor_attensor_attensor_attensor_double(void* input, void* grad_output, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_var, void* epsilon)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_batch_norm_backward(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_var).get())->get(), ((LanternObject<double>*)epsilon)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_input_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_backward_input(
        ((LanternObject<at::IntArrayRef>*)self_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_bias_attensor(void* grad_output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_backward_bias(
        ((LanternObject<at::Tensor>*)grad_output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_backward_weight_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_backward_weight(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* output_padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_convolution_transpose_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_input_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_transpose_backward_input(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_convolution_transpose_backward_weight_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_convolution_transpose_backward_weight(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self, void* weight, void* bias, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_depthwise_convolution(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_input_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* self_size, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_depthwise_convolution_backward_input(
        ((LanternObject<at::IntArrayRef>*)self_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool_stdarraybool(void* self, void* grad_output, void* weight, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_depthwise_convolution_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_depthwise_convolution_backward_weight_atintarrayref_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_intt_bool_bool(void* weight_size, void* grad_output, void* self, void* padding, void* stride, void* dilation, void* groups, void* benchmark, void* deterministic)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::miopen_depthwise_convolution_backward_weight(
        ((LanternObject<at::IntArrayRef>*)weight_size)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get(), ((LanternObject<bool>*)benchmark)->get(), ((LanternObject<bool>*)deterministic)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_attensor_attensorlist_intt_attensor_attensor_intt_intt_intt_bool_double_bool_bool_atintarrayref_attensor(void* input, void* weight, void* weight_stride0, void* hx, void* cx, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(cx).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<at::IntArrayRef>*)batch_sizes)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(dropout_state).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_miopen_rnn_backward_attensor_attensorlist_intt_attensor_attensor_attensor_attensor_attensor_attensor_attensor_intt_intt_intt_bool_double_bool_bool_atintarrayref_attensor_attensor_stdarraybool(void* input, void* weight, void* weight_stride0, void* weight_buf, void* hx, void* cx, void* output, void* grad_output, void* grad_hy, void* grad_cy, void* mode, void* hidden_size, void* num_layers, void* batch_first, void* dropout, void* train, void* bidirectional, void* batch_sizes, void* dropout_state, void* reserve, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::miopen_rnn_backward(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)weight)->get(), ((LanternObject<int64_t>*)weight_stride0)->get(), ((LanternObject<at::Tensor>*)weight_buf)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(cx).get())->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_output).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_cy).get())->get(), ((LanternObject<int64_t>*)mode)->get(), ((LanternObject<int64_t>*)hidden_size)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<at::IntArrayRef>*)batch_sizes)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(dropout_state).get())->get(), ((LanternObject<at::Tensor>*)reserve)->get(), ((LanternObject<std::array<bool,4>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mm(
        ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mm_out_attensor_attensor_attensor(void* out, void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_mm_attensor_attensor(void* sparse, void* dense)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_mm(
        ((LanternObject<at::Tensor>*)sparse)->get(), ((LanternObject<at::Tensor>*)dense)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sparse_matmul_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sparse_matmul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_mask_helper_attensor_attensor(void* t, void* mask_indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_mask_helper(
        ((LanternObject<at::Tensor>*)t)->get(), ((LanternObject<at::Tensor>*)mask_indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_attensor_intt_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().mode(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_attensor_attensor_attensor_intt_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mode_attensor_atdimname_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().mode(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mode_out_attensor_attensor_attensor_atdimname_bool(void* values, void* indices, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::mode_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mul(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mul_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mul_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mul_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mul(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mul__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mul_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multiply(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().multiply(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().multiply_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multiply_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multiply_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multiply(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().multiply(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multiply__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().multiply_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_attensor_attensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mv_attensor_attensor(void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mv(
        ((LanternObject<at::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mv_out_attensor_attensor_attensor(void* out, void* self, void* vec)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mvlgamma_attensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mvlgamma(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma_attensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mvlgamma(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_mvlgamma__attensor_intt(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().mvlgamma_(
        ((LanternObject<int64_t>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_copy_attensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::narrow_copy(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_copy_attensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().narrow_copy(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_copy_out_attensor_attensor_intt_intt_intt(void* out, void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::narrow_copy_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_attensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::narrow(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_attensor_intt_intt_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().narrow(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_narrow_attensor_intt_attensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::narrow(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_narrow_attensor_intt_attensor_intt(void* self, void* dim, void* start, void* length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().narrow(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)start)->get(), ((LanternObject<int64_t>*)length)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_attensor_attensor_attensor_attensor_attensor_bool_double_double(void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_out_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_bool_double_double(void* out, void* save_mean, void* save_invstd, void* input, void* weight, void* bias, void* running_mean, void* running_var, void* training, void* momentum, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)save_mean)->get(), ((LanternObject<at::Tensor>*)save_invstd)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_stats_attensor_double(void* input, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_stats(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<double>*)eps)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_attensor_attensor_attensor_attensor_attensor_double(void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::batch_norm_elemt(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_elemt_out_attensor_attensor_attensor_attensor_attensor_attensor_double(void* out, void* input, void* weight, void* bias, void* mean, void* invstd, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::batch_norm_elemt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_attensor_attensor_attensor_attensor_attensor_double_double_intt(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* count)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)count)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_gather_stats_with_counts_attensor_attensor_attensor_attensor_attensor_double_double_attensor(void* input, void* mean, void* invstd, void* running_mean, void* running_var, void* momentum, void* eps, void* counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_gather_stats_with_counts(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<at::Tensor>*)counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_native_batch_norm_backward_attensor_attensor_attensor_attensor_attensor_attensor_attensor_bool_double_stdarraybool(void* grad_out, void* input, void* weight, void* running_mean, void* running_var, void* save_mean, void* save_invstd, void* train, void* eps, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::native_batch_norm_backward(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(save_invstd).get())->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_reduce_attensor_attensor_attensor_attensor_attensor_bool_bool_bool(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* input_g, void* weight_g, void* bias_g)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_backward_reduce(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<bool>*)input_g)->get(), ((LanternObject<bool>*)weight_g)->get(), ((LanternObject<bool>*)bias_g)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_backward_elemt_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor(void* grad_out, void* input, void* mean, void* invstd, void* weight, void* mean_dy, void* mean_dy_xmu, void* count)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::batch_norm_backward_elemt(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)invstd)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<at::Tensor>*)mean_dy)->get(), ((LanternObject<at::Tensor>*)mean_dy_xmu)->get(), ((LanternObject<at::Tensor>*)count)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_batch_norm_update_stats_attensor_attensor_attensor_double(void* input, void* running_mean, void* running_var, void* momentum)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::batch_norm_update_stats(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_mean).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(running_var).get())->get(), ((LanternObject<double>*)momentum)->get())));
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

void* _lantern__nnpack_spatial_convolution_attensor_attensor_attensor_atintarrayref_atintarrayref(void* input, void* weight, void* bias, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_nnpack_spatial_convolution(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_attensor_attensor_attensor_atintarrayref_stdarraybool(void* input, void* grad_output, void* weight, void* padding, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_nnpack_spatial_convolution_backward(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_input_attensor_attensor_attensor_atintarrayref(void* input, void* grad_output, void* weight, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_nnpack_spatial_convolution_backward_input(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__nnpack_spatial_convolution_backward_weight_attensor_atintarrayref_attensor_atintarrayref(void* input, void* weightsize, void* grad_output, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_nnpack_spatial_convolution_backward_weight(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::IntArrayRef>*)weightsize)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_atintarrayref_atdimnamelist_attensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ones(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_atintarrayref_attensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ones(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_out_attensor_atintarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ones_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ones_like_attensor_attensoroptions_atmemoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ones_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pairwise_distance_attensor_attensor_double_double_bool(void* x1, void* x2, void* p, void* eps, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pairwise_distance(
        ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cdist_attensor_attensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cdist(
        ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__euclidean_dist_attensor_attensor(void* x1, void* x2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_euclidean_dist(
        ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_forward_attensor_attensor_double_intt(void* x1, void* x2, void* p, void* compute_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cdist_forward(
        ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<int64_t>>*)compute_mode)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cdist_backward_attensor_attensor_attensor_double_attensor(void* grad, void* x1, void* x2, void* p, void* cdist)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cdist_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<at::Tensor>*)cdist)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pdist_attensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pdist(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_forward_attensor_double(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_pdist_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pdist_backward_attensor_attensor_double_attensor(void* grad, void* self, void* p, void* pdist)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_pdist_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<at::Tensor>*)pdist)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cosine_similarity_attensor_attensor_intt_double(void* x1, void* x2, void* dim, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cosine_similarity(
        ((LanternObject<at::Tensor>*)x1)->get(), ((LanternObject<at::Tensor>*)x2)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<double>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_permute_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::permute(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_permute_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().permute(
        ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_attensor_atintarrayref_atintarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::movedim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)source)->get(), ((LanternObject<at::IntArrayRef>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_attensor_atintarrayref_atintarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().movedim(
        ((LanternObject<at::IntArrayRef>*)source)->get(), ((LanternObject<at::IntArrayRef>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_movedim_attensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::movedim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_movedim_attensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().movedim(
        ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_moveaxis_attensor_atintarrayref_atintarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::moveaxis(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)source)->get(), ((LanternObject<at::IntArrayRef>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_moveaxis_attensor_atintarrayref_atintarrayref(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().moveaxis(
        ((LanternObject<at::IntArrayRef>*)source)->get(), ((LanternObject<at::IntArrayRef>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_moveaxis_attensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::moveaxis(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_moveaxis_attensor_intt_intt(void* self, void* source, void* destination)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().moveaxis(
        ((LanternObject<int64_t>*)source)->get(), ((LanternObject<int64_t>*)destination)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_numpy_t_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().numpy_T(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pixel_shuffle_attensor_intt(void* self, void* upscale_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pixel_shuffle(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)upscale_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pixel_unshuffle_attensor_intt(void* self, void* downscale_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pixel_unshuffle(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)downscale_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_channel_shuffle_attensor_intt(void* self, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::channel_shuffle(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_pinned_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_pinned(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pin_memory_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pin_memory(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_pinverse_attensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pinverse(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pinverse_attensor_double(void* self, void* rcond)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pinverse(
        ((LanternObject<double>*)rcond)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_nll_loss_attensor_attensor_bool_bool_double_intt(void* input, void* target, void* log_input, void* full, void* eps, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::poisson_nll_loss(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<bool>*)log_input)->get(), ((LanternObject<bool>*)full)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rad2deg(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rad2deg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rad2deg_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rad2deg__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rad2deg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rad2deg_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rad2deg_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::deg2rad(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().deg2rad(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::deg2rad_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_deg2rad__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().deg2rad_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_deg2rad_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::deg2rad_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scalar_tensor_constatscalar_attensoroptions(void* s, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scalar_tensor(
        ((LanternObject<const at::Scalar &>*)s)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_atintarrayref_atdimnamelist_attensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_atintarrayref_atgenerator_atdimnamelist_attensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_atintarrayref_attensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_atintarrayref_atgenerator_attensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_attensor_atintarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_out_attensor_atintarrayref_atgenerator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rand_like_attensor_attensoroptions_atmemoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rand_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_atintarrayref_attensoroptions(void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_atintarrayref_atgenerator_attensoroptions(void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_atintarrayref_attensoroptions(void* low, void* high, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_intt_intt_atintarrayref_atgenerator_attensoroptions(void* low, void* high, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint(
        ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_attensor_intt_atintarrayref(void* out, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_attensor_intt_atintarrayref_atgenerator(void* out, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_attensor_intt_intt_atintarrayref(void* out, void* low, void* high, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_out_attensor_intt_intt_atintarrayref_atgenerator(void* out, void* low, void* high, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_attensor_intt_attensoroptions_atmemoryformat(void* self, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randint_like_attensor_intt_intt_attensoroptions_atmemoryformat(void* self, void* low, void* high, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randint_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)low)->get(), ((LanternObject<int64_t>*)high)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_atintarrayref_attensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_atintarrayref_atgenerator_attensoroptions(void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_atintarrayref_atdimnamelist_attensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_atintarrayref_atgenerator_atdimnamelist_attensoroptions(void* size, void* generator, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_attensor_atintarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_out_attensor_atintarrayref_atgenerator(void* out, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randn_like_attensor_attensoroptions_atmemoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randn_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_attensoroptions(void* n, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_intt_atgenerator_attensoroptions(void* n, void* generator, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randperm(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_attensor_intt(void* out, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randperm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_randperm_out_attensor_intt_atgenerator(void* out, void* n, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::randperm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_constatscalar_constatscalar_constatscalar_attensoroptions(void* start, void* end, void* step, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::range(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<const at::Scalar &>*)step)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_constatscalar_constatscalar_attensoroptions(void* start, void* end, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::range(
        ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_range_out_attensor_constatscalar_constatscalar_constatscalar(void* out, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::range_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)start)->get(), ((LanternObject<const at::Scalar &>*)end)->get(), ((LanternObject<const at::Scalar &>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ravel_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ravel(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ravel_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ravel(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reciprocal(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().reciprocal(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reciprocal_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reciprocal__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().reciprocal_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_reciprocal_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reciprocal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::neg(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().neg(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::neg_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_neg__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().neg_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_neg_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::neg_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::negative(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().negative(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::negative_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_negative__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().negative_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_negative_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::negative_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_attensor_atintarrayref(void* self, void* repeats)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().repeat(
        ((LanternObject<at::IntArrayRef>*)repeats)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_attensor(void* repeats)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::repeat_interleave(
        ((LanternObject<at::Tensor>*)repeats)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_attensor_attensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::repeat_interleave(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_attensor_attensor_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().repeat_interleave(
        ((LanternObject<at::Tensor>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_repeat_interleave_attensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::repeat_interleave(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_repeat_interleave_attensor_intt_intt(void* self, void* repeats, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().repeat_interleave(
        ((LanternObject<int64_t>*)repeats)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reshape_attensor_atintarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reshape(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_attensor_atintarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().reshape(
        ((LanternObject<at::IntArrayRef>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_reshape_attensor_atintarrayref(void* self, void* shape)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_mkldnn_reshape(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)shape)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_reshape_as_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().reshape_as(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_round_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::round(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().round(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::round_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_round__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().round_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_round_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::round_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_attensor_constatscalar_constatscalar_bool_atgenerator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu__attensor_constatscalar_constatscalar_bool_atgenerator(void* self, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_relu_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::relu(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().relu(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_relu__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::relu_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_relu__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().relu_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_relu6_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::relu6(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_relu6__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::relu6_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_attensor_attensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prelu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_attensor_attensor(void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().prelu(
        ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prelu_backward_attensor_attensor_attensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::prelu_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prelu_backward_attensor_attensor_attensor(void* grad_output, void* self, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)grad_output)->get().prelu_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gelu(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gelu_backward_attensor_attensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gelu_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_infinitely_differentiable_gelu_backward_attensor_attensor(void* grad, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::infinitely_differentiable_gelu_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_attensor_constatscalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardshrink(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_attensor_constatscalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().hardshrink(
        ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardshrink_backward_attensor_attensor_constatscalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardshrink_backward(
        ((LanternObject<at::Tensor>*)grad_out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hardshrink_backward_attensor_attensor_constatscalar(void* grad_out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)grad_out)->get().hardshrink_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rsqrt(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rsqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rsqrt_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rsqrt__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rsqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_rsqrt_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rsqrt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_attensor_atdimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::select(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_attensor_atdimname_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().select(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_attensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::select(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_select_attensor_intt_intt(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().select(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_select_backward_attensor_atintarrayref_intt_intt(void* grad, void* input_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::select_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_selu_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::selu(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_selu__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::selu_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_celu_attensor_constatscalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::celu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_celu__attensor_constatscalar(void* self, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::celu_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::silu(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::silu_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::silu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_silu_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::silu_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mish(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mish__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mish_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mish_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mish_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mish_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sigmoid(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sigmoid(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sigmoid_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sigmoid__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sigmoid_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sigmoid_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_attensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit_attensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logit(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit__attensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logit_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_logit__attensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().logit_(
        ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_out_attensor_attensor_double(void* out, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logit_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sin(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sin(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sin_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sin__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sin_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sin_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinc(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sinc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinc_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sinc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinc_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinc_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sinh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sinh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sinh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sinh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sinh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_detach_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::detach(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().detach(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_detach__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::detach_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_detach__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().detach_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_size_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_size_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::size(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_size_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().size(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_attensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slice(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)start)->get(), ((LanternObject<c10::optional<int64_t>>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slice_attensor_intt_intt_intt_intt(void* self, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().slice(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)start)->get(), ((LanternObject<c10::optional<int64_t>>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slice_backward_attensor_atintarrayref_intt_intt_intt_intt(void* grad, void* input_sizes, void* dim, void* start, void* end, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slice_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)start)->get(), ((LanternObject<int64_t>*)end)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slogdet_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slogdet(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_slogdet_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().slogdet(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_smm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::smm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_smm_attensor_attensor(void* self, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().smm(
        ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().softmax(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().softmax(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_attensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__softmax_backward_data_attensor_attensor_intt_attensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_softmax_backward_data(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_attensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unsafe_split(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_attensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().unsafe_split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_attensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::split(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_attensor_intt_intt(void* self, void* split_size, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().split(
        ((LanternObject<int64_t>*)split_size)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsafe_split_with_sizes_attensor_atintarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unsafe_split_with_sizes(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsafe_split_with_sizes_attensor_atintarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().unsafe_split_with_sizes(
        ((LanternObject<at::IntArrayRef>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_split_with_sizes_attensor_atintarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::split_with_sizes(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_split_with_sizes_attensor_atintarrayref_intt(void* self, void* split_sizes, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().split_with_sizes(
        ((LanternObject<at::IntArrayRef>*)split_sizes)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::hsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().hsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::hsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().hsplit(
        ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::vsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().vsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::vsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_vsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().vsplit(
        ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::dsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dsplit_attensor_intt(void* self, void* sections)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().dsplit(
        ((LanternObject<int64_t>*)sections)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::dsplit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dsplit_attensor_atintarrayref(void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().dsplit(
        ((LanternObject<at::IntArrayRef>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::squeeze(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::squeeze(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_squeeze_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::squeeze(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_squeeze__attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().squeeze_(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sspaddmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sspaddmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sspaddmm(
        ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sspaddmm_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sspaddmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_attensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::stack(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stack_out_attensor_attensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::stack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__stack_attensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_stack(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__stack_out_attensor_attensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_stack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hstack(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hstack_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hstack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::vstack(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vstack_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::vstack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dstack(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dstack_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dstack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stft_attensor_intt_intt_intt_attensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::stft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(window).get())->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stft_attensor_intt_intt_intt_attensor_bool_bool_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* normalized, void* onesided, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().stft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(window).get())->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(return_complex).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_istft_attensor_intt_intt_intt_attensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::istft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(window).get())->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_istft_attensor_intt_intt_intt_attensor_bool_bool_bool_intt_bool(void* self, void* n_fft, void* hop_length, void* win_length, void* window, void* center, void* normalized, void* onesided, void* length, void* return_complex)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().istft(
        ((LanternObject<int64_t>*)n_fft)->get(), ((LanternObject<c10::optional<int64_t>>*)hop_length)->get(), ((LanternObject<c10::optional<int64_t>>*)win_length)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(window).get())->get(), ((LanternObject<bool>*)center)->get(), ((LanternObject<bool>*)normalized)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(onesided).get())->get(), ((LanternObject<c10::optional<int64_t>>*)length)->get(), ((LanternObject<bool>*)return_complex)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().stride(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_stride_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::stride(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_stride_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().stride(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sum(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sum(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_attensor_atdimnamelist_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_attensor_atdimnamelist_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sum(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_attensor_attensor_atintarrayref_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sum_out_attensor_attensor_atdimnamelist_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nansum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nansum(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nansum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nansum_attensor_atintarrayref_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nansum(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nansum_out_attensor_attensor_atintarrayref_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nansum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sum_to_size_attensor_atintarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sum_to_size(
        ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sqrt(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sqrt(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sqrt_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sqrt__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sqrt_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sqrt_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sqrt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_square_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::square(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().square(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_square__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::square_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_square__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().square_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_square_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::square_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().std(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().std(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().std(
        ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_mean_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::std_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_attensor_attensor_atintarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_attensor_attensor_atintarrayref_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().std(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_attensor_attensor_atdimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_std_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().std(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_std_out_attensor_attensor_atdimnamelist_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::std_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().prod(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_attensor_intt_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_attensor_intt_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().prod(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_attensor_attensor_intt_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_attensor_atdimname_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_prod_attensor_atdimname_bool_atscalartype(void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().prod(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_prod_out_attensor_attensor_atdimname_bool_atscalartype(void* out, void* self, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::prod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_t_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::t(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().t(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_t__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().t_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tan(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tan(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tan_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tan__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tan_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tan_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tan_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tanh(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tanh(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tanh_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tanh__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tanh_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tanh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensordot_attensor_attensor_atintarrayref_atintarrayref(void* self, void* other, void* dims_self, void* dims_other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tensordot(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<at::IntArrayRef>*)dims_self)->get(), ((LanternObject<at::IntArrayRef>*)dims_other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tensordot_out_attensor_attensor_attensor_atintarrayref_atintarrayref(void* out, void* self, void* other, void* dims_self, void* dims_other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tensordot_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<at::IntArrayRef>*)dims_self)->get(), ((LanternObject<at::IntArrayRef>*)dims_other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_attensor_constatscalar_constatscalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::threshold(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold__attensor_constatscalar_constatscalar(void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::threshold_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_out_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* threshold, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::threshold_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_backward_out_attensor_attensor_attensor_constatscalar(void* grad_input, void* grad_output, void* self, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::threshold_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_threshold_backward_attensor_attensor_constatscalar(void* grad_output, void* self, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::threshold_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tile_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tile_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tile(
        ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().transpose(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_transpose_attensor_atdimname_atdimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim0)->get(), ((LanternObject<at::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose_attensor_atdimname_atdimname(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().transpose(
        ((LanternObject<at::Dimname>*)dim0)->get(), ((LanternObject<at::Dimname>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose_attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_mkldnn_transpose(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_transpose__attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().transpose_(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__mkldnn_transpose__attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_mkldnn_transpose_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_one_hot_attensor_intt(void* self, void* num_classes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::one_hot(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_classes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flip_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flip(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flip_attensor_atintarrayref(void* self, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flip(
        ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fliplr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fliplr(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fliplr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fliplr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_flipud_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flipud(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_flipud_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().flipud(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_roll_attensor_atintarrayref_atintarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::roll(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)shifts)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_roll_attensor_atintarrayref_atintarrayref(void* self, void* shifts, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().roll(
        ((LanternObject<at::IntArrayRef>*)shifts)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rot90_attensor_intt_atintarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rot90(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_rot90_attensor_intt_atintarrayref(void* self, void* k, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().rot90(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<at::IntArrayRef>*)dims)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_attensor_attensor_intt(void* y, void* x, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trapz(
        ((LanternObject<at::Tensor>*)y)->get(), ((LanternObject<at::Tensor>*)x)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trapz_attensor_double_intt(void* y, void* dx, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trapz(
        ((LanternObject<at::Tensor>*)y)->get(), ((LanternObject<double>*)dx)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__trilinear_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_intt(void* i1, void* i2, void* i3, void* expand1, void* expand2, void* expand3, void* sumdim, void* unroll_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_trilinear(
        ((LanternObject<at::Tensor>*)i1)->get(), ((LanternObject<at::Tensor>*)i2)->get(), ((LanternObject<at::Tensor>*)i3)->get(), ((LanternObject<at::IntArrayRef>*)expand1)->get(), ((LanternObject<at::IntArrayRef>*)expand2)->get(), ((LanternObject<at::IntArrayRef>*)expand3)->get(), ((LanternObject<at::IntArrayRef>*)sumdim)->get(), ((LanternObject<int64_t>*)unroll_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triplet_margin_loss_attensor_attensor_attensor_double_double_double_bool_intt(void* anchor, void* positive, void* negative, void* margin, void* p, void* eps, void* swap, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::triplet_margin_loss(
        ((LanternObject<at::Tensor>*)anchor)->get(), ((LanternObject<at::Tensor>*)positive)->get(), ((LanternObject<at::Tensor>*)negative)->get(), ((LanternObject<double>*)margin)->get(), ((LanternObject<double>*)p)->get(), ((LanternObject<double>*)eps)->get(), ((LanternObject<bool>*)swap)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trunc(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().trunc(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trunc_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trunc__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().trunc_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trunc_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trunc_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fix(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fix(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fix_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fix__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fix_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fix_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fix_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_type_as_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().type_as(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__has_compatible_shallow_copy_type_attensor_attensor(void* self, void* from)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::_has_compatible_shallow_copy_type(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)from)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__unique_attensor_bool_bool(void* self, void* sorted, void* return_inverse)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_attensor_intt_bool_bool_bool(void* self, void* dim, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_consecutive_attensor_bool_bool_intt(void* self, void* return_inverse, void* return_counts, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_consecutive(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_unique_dim_consecutive_attensor_intt_bool_bool(void* self, void* dim, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::unique_dim_consecutive(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unique2_attensor_bool_bool_bool(void* self, void* sorted, void* return_inverse, void* return_counts)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_unique2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)sorted)->get(), ((LanternObject<bool>*)return_inverse)->get(), ((LanternObject<bool>*)return_counts)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__unsafe_view_attensor_atintarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_unsafe_view(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unsqueeze_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::unsqueeze(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().unsqueeze(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unsqueeze__attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().unsqueeze_(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_vander_attensor_intt_bool(void* x, void* N, void* increasing)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::vander(
        ((LanternObject<at::Tensor>*)x)->get(), ((LanternObject<c10::optional<int64_t>>*)N)->get(), ((LanternObject<bool>*)increasing)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().var(
        ((LanternObject<bool>*)unbiased)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().var(
        ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().var(
        ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_attensor_attensor_atintarrayref_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_attensor_attensor_atintarrayref_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().var(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_attensor_attensor_atdimnamelist_bool_bool(void* out, void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_var_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().var(
        ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_out_attensor_attensor_atdimnamelist_intt_bool(void* out, void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::var_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_attensor_bool(void* self, void* unbiased)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)unbiased)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_attensor_atintarrayref_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_attensor_atintarrayref_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_attensor_atdimnamelist_bool_bool(void* self, void* dim, void* unbiased, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)unbiased)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_var_mean_attensor_atdimnamelist_intt_bool(void* self, void* dim, void* correction, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::var_mean(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<c10::optional<int64_t>>*)correction)->get(), ((LanternObject<bool>*)keepdim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_as_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().view_as(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_attensor_attensor_attensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::where(
        ((LanternObject<at::Tensor>*)condition)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_where_attensor_attensor_attensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)condition)->get().where(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_attensor_constatscalar_attensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::where(
        ((LanternObject<at::Tensor>*)condition)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_attensor_attensor_constatscalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::where(
        ((LanternObject<at::Tensor>*)condition)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_attensor_constatscalar_constatscalar(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::where(
        ((LanternObject<at::Tensor>*)condition)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_where_attensor(void* condition)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::where(
        ((LanternObject<at::Tensor>*)condition)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__s_where_attensor_attensor_attensor(void* condition, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_s_where(
        ((LanternObject<at::Tensor>*)condition)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_except_dim_attensor_intt_intt(void* v, void* pow, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm_except_dim(
        ((LanternObject<at::Tensor>*)v)->get(), ((LanternObject<int64_t>*)pow)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_attensor_attensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_weight_norm(
        ((LanternObject<at::Tensor>*)v)->get(), ((LanternObject<at::Tensor>*)g)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_attensor_attensor_intt(void* v, void* g, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface(
        ((LanternObject<at::Tensor>*)v)->get(), ((LanternObject<at::Tensor>*)g)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_cuda_interface_backward_attensor_attensor_attensor_attensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_cuda_interface_backward(
        ((LanternObject<at::Tensor>*)grad_w)->get(), ((LanternObject<at::Tensor>*)saved_v)->get(), ((LanternObject<at::Tensor>*)saved_g)->get(), ((LanternObject<at::Tensor>*)saved_norms)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__weight_norm_differentiable_backward_attensor_attensor_attensor_attensor_intt(void* grad_w, void* saved_v, void* saved_g, void* saved_norms, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_weight_norm_differentiable_backward(
        ((LanternObject<at::Tensor>*)grad_w)->get(), ((LanternObject<at::Tensor>*)saved_v)->get(), ((LanternObject<at::Tensor>*)saved_g)->get(), ((LanternObject<at::Tensor>*)saved_norms)->get(), ((LanternObject<int64_t>*)dim)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_atintarrayref_atdimnamelist_attensoroptions(void* size, void* names, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::zeros(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::DimnameList>>*)optional<at::DimnameList>(names).get())->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_atintarrayref_attensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::zeros(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_out_attensor_atintarrayref(void* out, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::zeros_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zeros_like_attensor_attensoroptions_atmemoryformat(void* self, void* options, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::zeros_like(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_grad_attensor_attensor(void* self, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_standard_gamma_grad(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__standard_gamma_attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_standard_gamma(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__dirichlet_grad_attensor_attensor_attensor(void* x, void* alpha, void* total)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_dirichlet_grad(
        ((LanternObject<at::Tensor>*)x)->get(), ((LanternObject<at::Tensor>*)alpha)->get(), ((LanternObject<at::Tensor>*)total)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sample_dirichlet_attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sample_dirichlet(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_poisson_attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::poisson(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_binomial_attensor_attensor_atgenerator(void* count, void* prob, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::binomial(
        ((LanternObject<at::Tensor>*)count)->get(), ((LanternObject<at::Tensor>*)prob)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_attensor_constatscalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::native_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_native_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::native_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sum(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_attensor_atintarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_attensor_atintarrayref_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_sum_backward_attensor_attensor_atintarrayref(void* grad, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_sum_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_attensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_softmax_backward_data_attensor_attensor_intt_attensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_softmax_backward_data(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_attensor_intt_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_attensor_atdimname_atscalartype(void* self, void* dim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_attensor_intt_bool(void* self, void* dim, void* half_to_float)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_log_softmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)half_to_float)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_log_softmax_backward_data_attensor_attensor_intt_attensor(void* grad_output, void* output, void* dim, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_log_softmax_backward_data(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar_atscalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar_atscalartype(void* self, void* p, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<const at::Scalar &>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar_atintarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar_atintarrayref_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_attensor_attensor_constatscalar_atintarrayref_bool_atscalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_attensor_attensor_constatscalar_atintarrayref_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar_atdimnamelist_bool_atscalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar_atdimnamelist_bool_atscalartype(void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_attensor_constatscalar_atdimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_norm_attensor_constatscalar_atdimnamelist_bool(void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().norm(
        ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_attensor_attensor_constatscalar_atdimnamelist_bool_atscalartype(void* out, void* self, void* p, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_norm_out_attensor_attensor_constatscalar_atdimnamelist_bool(void* out, void* self, void* p, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get(), ((LanternObject<at::DimnameList>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frexp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::frexp(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_frexp_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().frexp(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_frexp_out_attensor_attensor_attensor(void* mantissa, void* exponent, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::frexp_out(
        ((LanternObject<at::Tensor>*)mantissa)->get(), ((LanternObject<at::Tensor>*)exponent)->get(), ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frobenius_norm(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frobenius_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_frobenius_norm_out_attensor_attensor_atintarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::frobenius_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_attensor_bool(void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nuclear_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_attensor_attensor_bool(void* out, void* self, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_attensor_atintarrayref_bool(void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nuclear_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nuclear_norm_out_attensor_attensor_atintarrayref_bool(void* out, void* self, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nuclear_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_clone_attensor_atmemoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::clone(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_clone_attensor_atmemoryformat(void* self, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().clone(
        ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_positive_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::positive(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_positive_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().positive(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_resize_as__attensor_attensor_atmemoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::resize_as_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)the_template)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_resize_as__attensor_attensor_atmemoryformat(void* self, void* the_template, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().resize_as_(
        ((LanternObject<at::Tensor>*)the_template)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_resize_as_sparse__attensor_attensor(void* self, void* the_template)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::resize_as_sparse_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)the_template)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_zero__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::zero_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_zero__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().zero_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_out_attensor_attensor_attensor_constatscalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sub_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sub(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sub(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sub_(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sub_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sub(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sub(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sub__attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sub_(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_out_attensor_attensor_attensor_constatscalar(void* out, void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::subtract_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::subtract(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().subtract(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().subtract_(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_subtract_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::subtract(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().subtract(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_subtract__attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().subtract_(
        ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rsub(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_out_attensor_attensor_attensor(void* out, void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::heaviside_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_heaviside_attensor_attensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::heaviside(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside_attensor_attensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().heaviside(
        ((LanternObject<at::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_heaviside__attensor_attensor(void* self, void* values)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().heaviside_(
        ((LanternObject<at::Tensor>*)values)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rsub_attensor_constatscalar_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rsub(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_addmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* sparse, void* dense, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_addmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)sparse)->get(), ((LanternObject<at::Tensor>*)dense)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addmm(
        ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addmm__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* mat1, void* mat2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addmm_(
        ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_csr_tensor_attensor_attensor_attensor_atintarrayref_attensoroptions(void* crow_indices, void* col_indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_csr_tensor(
        ((LanternObject<at::Tensor>*)crow_indices)->get(), ((LanternObject<at::Tensor>*)col_indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_csr_tensor_attensor_attensor_attensor_attensoroptions(void* crow_indices, void* col_indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_csr_tensor(
        ((LanternObject<at::Tensor>*)crow_indices)->get(), ((LanternObject<at::Tensor>*)col_indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_atintarrayref_attensoroptions(void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_attensor_attensor_attensoroptions(void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sparse_coo_tensor_attensor_attensor_atintarrayref_attensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sparse_coo_tensor(
        ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_unsafe_attensor_attensor_atintarrayref_attensoroptions(void* indices, void* values, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_coo_tensor_unsafe(
        ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__validate_sparse_coo_tensor_args_attensor_attensor_atintarrayref(void* indices, void* values, void* size)
{
  LANTERN_FUNCTION_START
    torch::_validate_sparse_coo_tensor_args(((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::IntArrayRef>*)size)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_intt_intt_atintarrayref_attensoroptions(void* sparse_dim, void* dense_dim, void* size, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_coo_tensor_with_dims(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__sparse_coo_tensor_with_dims_and_tensors_intt_intt_atintarrayref_attensor_attensor_attensoroptions(void* sparse_dim, void* dense_dim, void* size, void* indices, void* values, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_sparse_coo_tensor_with_dims_and_tensors(
        ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize__attensor_atintarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sparse_resize_(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_resize_and_clear__attensor_atintarrayref_intt_intt(void* self, void* size, void* sparse_dim, void* dense_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sparse_resize_and_clear_(
        ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<int64_t>*)sparse_dim)->get(), ((LanternObject<int64_t>*)dense_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_mask_attensor_attensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sparse_mask(
        ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_dense_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to_dense(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_to_dense_backward_attensor_attensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::to_dense_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sparse_dim_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().sparse_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimi_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get()._dimI(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dense_dim_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().dense_dim(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__dimv_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get()._dimV(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__nnz_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get()._nnz(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_coalesce_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().coalesce(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__coalesce_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_coalesce(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_coalesced_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_coalesced(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__indices_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get()._indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__values_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get()._values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor__coalesced__attensor_bool(void* self, void* coalesced)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get()._coalesced_(
        ((LanternObject<bool>*)coalesced)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_indices_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_values_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().values(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_crow_indices_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().crow_indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_col_indices_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().col_indices(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_out_attensor_attensor_attensor(void* out, void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hspmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hspmm_attensor_attensor(void* mat1, void* mat2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hspmm(
        ((LanternObject<at::Tensor>*)mat1)->get(), ((LanternObject<at::Tensor>*)mat2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_copy_sparse_to_sparse__attensor_attensor_bool(void* self, void* src, void* non_blocking)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::copy_sparse_to_sparse_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)src)->get(), ((LanternObject<bool>*)non_blocking)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unbind(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().unbind(
        ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unbind_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unbind(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unbind_attensor_atdimname(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().unbind(
        ((LanternObject<at::Dimname>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_attensor_intt(void* self, void* sparse_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to_sparse(
        ((LanternObject<int64_t>*)sparse_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_sparse_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to_sparse(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_mkldnn_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to_mkldnn(
        ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv2d_weight_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_reorder_conv2d_weight(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_reorder_conv3d_weight_attensor_atintarrayref_atintarrayref_atintarrayref_intt(void* self, void* padding, void* stride, void* dilation, void* groups)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_reorder_conv3d_weight(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<int64_t>*)groups)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_to_mkldnn_backward_attensor_attensor(void* grad, void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::to_mkldnn_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_attensor_double_intt_atscalartype(void* self, void* scale, void* zero_point, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantize_per_tensor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_tensor_attensorlist_attensor_attensor_atscalartype(void* tensors, void* scales, void* zero_points, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::quantize_per_tensor(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::Tensor>*)scales)->get(), ((LanternObject<at::Tensor>*)zero_points)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantize_per_channel_attensor_attensor_attensor_intt_atscalartype(void* self, void* scales, void* zero_points, void* axis, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantize_per_channel(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scales)->get(), ((LanternObject<at::Tensor>*)zero_points)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dequantize_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dequantize(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dequantize_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().dequantize(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_dequantize_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::dequantize(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_q_scale_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<double>(torch::q_scale(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_scale_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<double>(((LanternObject<at::Tensor>*)self)->get().q_scale(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_zero_point_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_zero_point(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_zero_point_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().q_zero_point(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_scales_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::q_per_channel_scales(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_scales_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().q_per_channel_scales(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_zero_points_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::q_per_channel_zero_points(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_zero_points_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().q_per_channel_zero_points(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_q_per_channel_axis_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(torch::q_per_channel_axis(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_q_per_channel_axis_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<int64_t>(((LanternObject<at::Tensor>*)self)->get().q_per_channel_axis(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_int_repr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::int_repr(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_int_repr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().int_repr(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_tensor_quantized_tensor_attensor_double_intt(void* self, void* scale, void* zero_point)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_make_per_tensor_quantized_tensor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__make_per_channel_quantized_tensor_attensor_attensor_attensor_intt(void* self, void* scale, void* zero_point, void* axis)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_make_per_channel_quantized_tensor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qscheme_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::QScheme>(((LanternObject<at::Tensor>*)self)->get().qscheme(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_attensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fake_quantize_per_tensor_affine(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_cachemask_attensor_double_intt_intt_intt(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fake_quantize_per_tensor_affine_cachemask(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)scale)->get(), ((LanternObject<int64_t>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_tensor_affine_cachemask_backward_attensor_attensor(void* grad, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fake_quantize_per_tensor_affine_cachemask_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_attensor_attensor_attensor_intt_intt_double(void* self, void* scale, void* zero_point, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fake_quantize_learnable_per_tensor_affine(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_tensor_affine_backward_attensor_attensor_attensor_attensor_intt_intt_double(void* grad, void* self, void* scale, void* zero_point, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_tensor_affine_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_attensor_attensor_attensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fake_quantize_per_channel_affine(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_cachemask_attensor_attensor_attensor_intt_intt_intt(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fake_quantize_per_channel_affine_cachemask(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fake_quantize_per_channel_affine_cachemask_backward_attensor_attensor(void* grad, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fake_quantize_per_channel_affine_cachemask_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_attensor_attensor_attensor_intt_intt_intt_double(void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_fake_quantize_learnable_per_channel_affine(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__fake_quantize_learnable_per_channel_affine_backward_attensor_attensor_attensor_attensor_intt_intt_intt_double(void* grad, void* self, void* scale, void* zero_point, void* axis, void* quant_min, void* quant_max, void* grad_factor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_fake_quantize_learnable_per_channel_affine_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)scale)->get(), ((LanternObject<at::Tensor>*)zero_point)->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<int64_t>*)quant_min)->get(), ((LanternObject<int64_t>*)quant_max)->get(), ((LanternObject<double>*)grad_factor)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__choose_qparams_per_tensor_attensor_bool(void* self, void* reduce_range)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_choose_qparams_per_tensor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)reduce_range)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__saturate_weight_to_fp16_attensor(void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_saturate_weight_to_fp16(
        ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_choose_qparams_optimized_attensor_intt_intt_double_intt(void* input, void* numel, void* n_bins, void* ratio, void* bit_width)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::choose_qparams_optimized(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<int64_t>*)numel)->get(), ((LanternObject<int64_t>*)n_bins)->get(), ((LanternObject<double>*)ratio)->get(), ((LanternObject<int64_t>*)bit_width)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_attensor_attensoroptions_bool_bool_atmemoryformat(void* self, void* options, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to(
        ((LanternObject<at::TensorOptions>*)options)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_attensor_atdevice_atscalartype_bool_bool_atmemoryformat(void* self, void* device, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to(
        ((LanternObject<at::Device>*)device)->get(), ((LanternObject<at::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_attensor_atscalartype_bool_bool_atmemoryformat(void* self, void* dtype, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to(
        ((LanternObject<at::ScalarType>*)dtype)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_to_attensor_attensor_bool_bool_atmemoryformat(void* self, void* other, void* non_blocking, void* copy, void* memory_format)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().to(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<bool>*)non_blocking)->get(), ((LanternObject<bool>*)copy)->get(), ((LanternObject<c10::optional<at::MemoryFormat>>*)optional<at::MemoryFormat>(memory_format).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_meshgrid_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::meshgrid(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cartesian_prod_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cartesian_prod(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_combinations_attensor_intt_bool(void* self, void* r, void* with_replacement)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::combinations(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)r)->get(), ((LanternObject<bool>*)with_replacement)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_item_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<const at::Scalar &>(((LanternObject<at::Tensor>*)self)->get().item(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_attensor_attensor(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::ScalarType>(torch::result_type(
        ((LanternObject<at::Tensor>*)tensor)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_attensor_constatscalar(void* tensor, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::ScalarType>(torch::result_type(
        ((LanternObject<at::Tensor>*)tensor)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_constatscalar_attensor(void* scalar, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::ScalarType>(torch::result_type(
        ((LanternObject<const at::Scalar &>*)scalar)->get(), ((LanternObject<at::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_result_type_constatscalar_constatscalar(void* scalar1, void* scalar2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::ScalarType>(torch::result_type(
        ((LanternObject<const at::Scalar &>*)scalar1)->get(), ((LanternObject<const at::Scalar &>*)scalar2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_can_cast_atscalartype_atscalartype(void* from, void* to)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::can_cast(
        ((LanternObject<at::ScalarType>*)from)->get(), ((LanternObject<at::ScalarType>*)to)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_promote_types_atscalartype_atscalartype(void* type1, void* type2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::ScalarType>(torch::promote_types(
        ((LanternObject<at::ScalarType>*)type1)->get(), ((LanternObject<at::ScalarType>*)type2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__local_scalar_dense_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<const at::Scalar &>(torch::_local_scalar_dense(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_attensor_attensor_attensor_attensor_attensor(void* input_gates, void* hidden_gates, void* cx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell(
        ((LanternObject<at::Tensor>*)input_gates)->get(), ((LanternObject<at::Tensor>*)hidden_gates)->get(), ((LanternObject<at::Tensor>*)cx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_lstm_cell_backward_attensor_attensor_attensor_attensor_attensor_bool(void* grad_hy, void* grad_cy, void* cx, void* cy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_lstm_cell_backward(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_cy).get())->get(), ((LanternObject<at::Tensor>*)cx)->get(), ((LanternObject<at::Tensor>*)cy)->get(), ((LanternObject<at::Tensor>*)workspace)->get(), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_lstm_cell_backward_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor(void* grad_hy, void* grad_cy, void* input_gates, void* hidden_gates, void* input_bias, void* hidden_bias, void* cx, void* cy)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_lstm_cell_backward(
        ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_hy).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(grad_cy).get())->get(), ((LanternObject<at::Tensor>*)input_gates)->get(), ((LanternObject<at::Tensor>*)hidden_gates)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(hidden_bias).get())->get(), ((LanternObject<at::Tensor>*)cx)->get(), ((LanternObject<at::Tensor>*)cy)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_attensor_attensor_attensor_attensor_attensor(void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell(
        ((LanternObject<at::Tensor>*)input_gates)->get(), ((LanternObject<at::Tensor>*)hidden_gates)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_fused_gru_cell_backward_attensor_attensor_bool(void* grad_hy, void* workspace, void* has_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_fused_gru_cell_backward(
        ((LanternObject<at::Tensor>*)grad_hy)->get(), ((LanternObject<at::Tensor>*)workspace)->get(), ((LanternObject<bool>*)has_bias)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__thnn_differentiable_gru_cell_backward_attensor_attensor_attensor_attensor_attensor_attensor(void* grad_hy, void* input_gates, void* hidden_gates, void* hx, void* input_bias, void* hidden_bias)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_thnn_differentiable_gru_cell_backward(
        ((LanternObject<at::Tensor>*)grad_hy)->get(), ((LanternObject<at::Tensor>*)input_gates)->get(), ((LanternObject<at::Tensor>*)hidden_gates)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(input_bias).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(hidden_bias).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_attensor_attensorlist_attensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_attensor_attensor_attensorlist_attensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<at::TensorList>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_attensor_attensor_attensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_attensor_attensor_attensor_attensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::gru(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_attensor_attensor_attensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_attensor_attensor_attensor_attensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_tanh(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_attensor_attensor_attensorlist_bool_intt_double_bool_bool_bool(void* input, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_attensor_attensor_attensor_attensorlist_bool_intt_double_bool_bool(void* data, void* batch_sizes, void* hx, void* params, void* has_biases, void* num_layers, void* dropout, void* train, void* bidirectional)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::rnn_relu(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::TensorList>*)params)->get(), ((LanternObject<bool>*)has_biases)->get(), ((LanternObject<int64_t>*)num_layers)->get(), ((LanternObject<double>*)dropout)->get(), ((LanternObject<bool>*)train)->get(), ((LanternObject<bool>*)bidirectional)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstm_cell_attensor_attensorlist_attensor_attensor_attensor_attensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstm_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_hh).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_gru_cell_attensor_attensor_attensor_attensor_attensor_attensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gru_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_tanh_cell_attensor_attensor_attensor_attensor_attensor_attensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rnn_tanh_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rnn_relu_cell_attensor_attensor_attensor_attensor_attensor_attensor(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rnn_relu_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_ih).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(b_hh).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_lstm_cell_attensor_attensorlist_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_constatscalar_constatscalar_constatscalar_constatscalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::quantized_lstm_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::TensorList>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<at::Tensor>*)b_ih)->get(), ((LanternObject<at::Tensor>*)b_hh)->get(), ((LanternObject<at::Tensor>*)packed_ih)->get(), ((LanternObject<at::Tensor>*)packed_hh)->get(), ((LanternObject<at::Tensor>*)col_offsets_ih)->get(), ((LanternObject<at::Tensor>*)col_offsets_hh)->get(), ((LanternObject<const at::Scalar &>*)scale_ih)->get(), ((LanternObject<const at::Scalar &>*)scale_hh)->get(), ((LanternObject<const at::Scalar &>*)zero_point_ih)->get(), ((LanternObject<const at::Scalar &>*)zero_point_hh)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_gru_cell_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_constatscalar_constatscalar_constatscalar_constatscalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_gru_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<at::Tensor>*)b_ih)->get(), ((LanternObject<at::Tensor>*)b_hh)->get(), ((LanternObject<at::Tensor>*)packed_ih)->get(), ((LanternObject<at::Tensor>*)packed_hh)->get(), ((LanternObject<at::Tensor>*)col_offsets_ih)->get(), ((LanternObject<at::Tensor>*)col_offsets_hh)->get(), ((LanternObject<const at::Scalar &>*)scale_ih)->get(), ((LanternObject<const at::Scalar &>*)scale_hh)->get(), ((LanternObject<const at::Scalar &>*)zero_point_ih)->get(), ((LanternObject<const at::Scalar &>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_relu_cell_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_constatscalar_constatscalar_constatscalar_constatscalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_rnn_relu_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<at::Tensor>*)b_ih)->get(), ((LanternObject<at::Tensor>*)b_hh)->get(), ((LanternObject<at::Tensor>*)packed_ih)->get(), ((LanternObject<at::Tensor>*)packed_hh)->get(), ((LanternObject<at::Tensor>*)col_offsets_ih)->get(), ((LanternObject<at::Tensor>*)col_offsets_hh)->get(), ((LanternObject<const at::Scalar &>*)scale_ih)->get(), ((LanternObject<const at::Scalar &>*)scale_hh)->get(), ((LanternObject<const at::Scalar &>*)zero_point_ih)->get(), ((LanternObject<const at::Scalar &>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantized_rnn_tanh_cell_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_attensor_constatscalar_constatscalar_constatscalar_constatscalar(void* input, void* hx, void* w_ih, void* w_hh, void* b_ih, void* b_hh, void* packed_ih, void* packed_hh, void* col_offsets_ih, void* col_offsets_hh, void* scale_ih, void* scale_hh, void* zero_point_ih, void* zero_point_hh)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantized_rnn_tanh_cell(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)hx)->get(), ((LanternObject<at::Tensor>*)w_ih)->get(), ((LanternObject<at::Tensor>*)w_hh)->get(), ((LanternObject<at::Tensor>*)b_ih)->get(), ((LanternObject<at::Tensor>*)b_hh)->get(), ((LanternObject<at::Tensor>*)packed_ih)->get(), ((LanternObject<at::Tensor>*)packed_hh)->get(), ((LanternObject<at::Tensor>*)col_offsets_ih)->get(), ((LanternObject<at::Tensor>*)col_offsets_hh)->get(), ((LanternObject<const at::Scalar &>*)scale_ih)->get(), ((LanternObject<const at::Scalar &>*)scale_hh)->get(), ((LanternObject<const at::Scalar &>*)zero_point_ih)->get(), ((LanternObject<const at::Scalar &>*)zero_point_hh)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_attensor_attensor_bool(void* input, void* lengths, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pack_padded_sequence(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)lengths)->get(), ((LanternObject<bool>*)batch_first)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__pack_padded_sequence_backward_attensor_atintarrayref_attensor_bool(void* grad, void* input_size, void* batch_sizes, void* batch_first)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_pack_padded_sequence_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<bool>*)batch_first)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__pad_packed_sequence_attensor_attensor_bool_constatscalar_intt(void* data, void* batch_sizes, void* batch_first, void* padding_value, void* total_length)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_pad_packed_sequence(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<at::Tensor>*)batch_sizes)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<const at::Scalar &>*)padding_value)->get(), ((LanternObject<int64_t>*)total_length)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__attensor_atstorage(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().set_(
        ((LanternObject<at::Storage>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__attensor_atstorage_intt_atintarrayref_atintarrayref(void* self, void* source, void* storage_offset, void* size, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().set_(
        ((LanternObject<at::Storage>*)source)->get(), ((LanternObject<int64_t>*)storage_offset)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__attensor_attensor(void* self, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().set_(
        ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_set__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().set_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_is_set_to_attensor_attensor(void* self, void* tensor)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().is_set_to(
        ((LanternObject<at::Tensor>*)tensor)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__attensor_attensor_constatscalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_fill_(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_attensor_attensor_constatscalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_attensor_attensor_constatscalar(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_fill(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill__attensor_attensor_attensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_fill_(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_fill_attensor_attensor_attensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_fill_attensor_attensor_attensor(void* self, void* mask, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_fill(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter__attensor_attensor_attensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_scatter_(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_scatter_attensor_attensor_attensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_scatter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_scatter_attensor_attensor_attensor(void* self, void* mask, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_scatter(
        ((LanternObject<at::Tensor>*)mask)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_attensor_atintarrayref(void* self, void* size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().view(
        ((LanternObject<at::IntArrayRef>*)size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_view_attensor_atscalartype(void* self, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().view(
        ((LanternObject<at::ScalarType>*)dtype)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_put__attensor_attensor_attensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().put_(
        ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_put_attensor_attensor_attensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::put(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_put_attensor_attensor_attensor_bool(void* self, void* index, void* source, void* accumulate)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().put(
        ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<bool>*)accumulate)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_add_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add__attensor_intt_attensor_attensor_constatscalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_add_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_add(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_attensor_intt_attensor_attensor_constatscalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_attensor_intt_attensor_attensor_constatscalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_add(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_add_attensor_atdimname_attensor_attensor_constatscalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_add_attensor_atdimname_attensor_attensor_constatscalar(void* self, void* dim, void* index, void* source, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_add(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__attensor_atdimname_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill_(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill__attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill_(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_attensor_atdimname_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_attensor_atdimname_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_fill_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_fill(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_fill_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_fill(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_attensor_intt_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_attensor_atdimname_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_attensor_atdimname_attensor_constatscalar(void* self, void* dim, void* index, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__attensor_intt_attensor_attensor_stdstring(void* self, void* dim, void* index, void* src, void* reduce)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get(), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter__attensor_intt_attensor_constatscalar_stdstring(void* self, void* dim, void* index, void* value, void* reduce)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<const at::Scalar &>*)value)->get(), ((LanternObject<std::string>*)reduce)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_add_(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter_add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_add(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_scatter_add_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::scatter_add(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_scatter_add_attensor_atdimname_attensor_attensor(void* self, void* dim, void* index, void* src)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().scatter_add(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)src)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().eq_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().eq_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_and_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_and_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_and(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_and(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_and_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_and(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_and(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_and_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_and__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_and_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___and___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__and__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__and__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___and___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__and__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___and___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__and__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__iand__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___iand___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__iand__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_or_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_or_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_or(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_or(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_or_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_or(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_or(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_or_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_or__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_or_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___or___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__or__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__or__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___or___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__or__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___or___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__or__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ior__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ior___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ior__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_xor_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_xor_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_xor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_xor(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bitwise_xor_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bitwise_xor(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_xor(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_xor_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_bitwise_xor__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().bitwise_xor_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__xor__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__xor__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___xor___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__xor__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___xor___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__xor__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ixor__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ixor___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ixor__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__lshift__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__lshift__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___lshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__lshift__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___lshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__lshift__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ilshift__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___ilshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__ilshift__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__rshift__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__rshift__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern___rshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::__rshift__(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___rshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__rshift__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__irshift__(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor___irshift___attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().__irshift__(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril__attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tril_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu__attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().triu_(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().digamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm__attensor_constatscalar_intt_constatscalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().renorm_(
        ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<const at::Scalar &>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__attensor_attensor_constatscalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lerp_(
        ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<const at::Scalar &>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp__attensor_attensor_attensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lerp_(
        ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmod_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmod_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().remainder_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().remainder_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm__attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addbmm_(
        ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addbmm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addbmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addbmm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addbmm_attensor_attensor_attensor_constatscalar_constatscalar(void* self, void* batch1, void* batch2, void* beta, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addbmm(
        ((LanternObject<at::Tensor>*)batch1)->get(), ((LanternObject<at::Tensor>*)batch2)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv__attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addcdiv_(
        ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__attensor_intt_intt_atgenerator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().random_(
        ((LanternObject<int64_t>*)from)->get(), ((LanternObject<c10::optional<int64_t>>*)to)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__attensor_intt_atgenerator(void* self, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().random_(
        ((LanternObject<int64_t>*)to)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_random__attensor_atgenerator(void* self, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().random_(
        ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_uniform__attensor_double_double_atgenerator(void* self, void* from, void* to, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().uniform_(
        ((LanternObject<double>*)from)->get(), ((LanternObject<double>*)to)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cauchy__attensor_double_double_atgenerator(void* self, void* median, void* sigma, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cauchy_(
        ((LanternObject<double>*)median)->get(), ((LanternObject<double>*)sigma)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_log_normal__attensor_double_double_atgenerator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().log_normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_exponential__attensor_double_atgenerator(void* self, void* lambd, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().exponential_(
        ((LanternObject<double>*)lambd)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geometric__attensor_double_atgenerator(void* self, void* p, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().geometric_(
        ((LanternObject<double>*)p)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_out_attensor_attensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diag_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diag(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_diag_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().diag(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_diag_backward_attensor_atintarrayref_intt(void* grad, void* input_sizes, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::diag_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)input_sizes)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_out_attensor_attensor_attensor_intt(void* out, void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cross_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_attensor_attensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cross(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cross_attensor_attensor_intt(void* self, void* other, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cross(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_out_attensor_attensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::triu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::triu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triu_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().triu(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_out_attensor_attensor_intt(void* out, void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tril_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tril(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_tril_attensor_intt(void* self, void* diagonal)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().tril(
        ((LanternObject<int64_t>*)diagonal)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tril_indices_intt_intt_intt_attensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tril_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_triu_indices_intt_intt_intt_attensoroptions(void* row, void* col, void* offset, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::triu_indices(
        ((LanternObject<int64_t>*)row)->get(), ((LanternObject<int64_t>*)col)->get(), ((LanternObject<int64_t>*)offset)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trace(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_trace_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().trace(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_trace_backward_attensor_atintarrayref(void* grad, void* sizes)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::trace_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)sizes)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ne_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ne(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ne(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ne_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ne_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ne(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ne(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ne_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ne__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ne_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::not_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::not_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().not_equal(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::not_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_not_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::not_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().not_equal(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().not_equal_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_not_equal__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().not_equal_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eq_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eq(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().eq(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eq_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_eq_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::eq(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eq_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().eq(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ge_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ge(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ge(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ge_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ge_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ge(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ge(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ge_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ge__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ge_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_equal(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_equal(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_equal_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_equal__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_equal_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::le_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::le(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().le(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::le_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_le_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::le(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().le(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().le_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_le__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().le_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_equal(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_equal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_equal(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_equal_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_equal__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_equal_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gt(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gt(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gt_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gt(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gt(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gt_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gt__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gt_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_greater_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::greater(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_greater__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().greater_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lt(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lt(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lt_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lt_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lt(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lt(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lt_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lt__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lt_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_less_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::less(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_less__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().less_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_out_attensor_attensor_attensor(void* out, void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::take_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_attensor_attensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::take(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_take_attensor_attensor(void* self, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().take(
        ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_along_dim_out_attensor_attensor_attensor_intt(void* out, void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::take_along_dim_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_take_along_dim_attensor_attensor_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::take_along_dim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_take_along_dim_attensor_attensor_intt(void* self, void* indices, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().take_along_dim(
        ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_attensor_attensor_intt_attensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_select_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_attensor_intt_attensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_select(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_attensor_intt_attensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_select(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_out_attensor_attensor_atdimname_attensor(void* out, void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_select_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_attensor_atdimname_attensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_select(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_index_select_attensor_atdimname_attensor(void* self, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().index_select(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_index_select_backward_attensor_atintarrayref_intt_attensor(void* grad, void* self_sizes, void* dim, void* index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::index_select_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::IntArrayRef>*)self_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_out_attensor_attensor_attensor(void* out, void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_select_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_attensor_attensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_select(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_masked_select_attensor_attensor(void* self, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().masked_select(
        ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_masked_select_backward_attensor_attensor_attensor(void* grad, void* input, void* mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::masked_select_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)mask)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nonzero_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nonzero(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nonzero(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_nonzero_numpy_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::nonzero_numpy(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nonzero_numpy_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(((LanternObject<at::Tensor>*)self)->get().nonzero_numpy(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_attensor_attensor_intt_attensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gather_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_attensor_intt_attensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gather(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_attensor_intt_attensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gather(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_backward_attensor_attensor_intt_attensor_bool(void* grad, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gather_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_out_attensor_attensor_atdimname_attensor_bool(void* out, void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gather_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_gather_attensor_atdimname_attensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::gather(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_gather_attensor_atdimname_attensor_bool(void* self, void* dim, void* index, void* sparse_grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().gather(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<bool>*)sparse_grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__gather_sparse_backward_attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* grad)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_gather_sparse_backward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)grad)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_out_attensor_attensor_attensor_attensor_constatscalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addcmul_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcmul_attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addcmul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul_attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addcmul(
        ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcmul__attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addcmul_(
        ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_out_attensor_attensor_attensor_attensor_constatscalar(void* out, void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addcdiv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_addcdiv_attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::addcdiv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_addcdiv_attensor_attensor_attensor_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().addcdiv(
        ((LanternObject<at::Tensor>*)tensor1)->get(), ((LanternObject<at::Tensor>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cross_entropy_loss_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cross_entropy_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_out_attensor_attensor_attensor_attensor(void* X, void* qr, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq_out(
        ((LanternObject<at::Tensor>*)X)->get(), ((LanternObject<at::Tensor>*)qr)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lstsq_attensor_attensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lstsq(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lstsq_attensor_attensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().lstsq(
        ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_out_attensor_attensor_attensor_attensor_bool_bool_bool(void* X, void* M, void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve_out(
        ((LanternObject<at::Tensor>*)X)->get(), ((LanternObject<at::Tensor>*)M)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_triangular_solve_attensor_attensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::triangular_solve(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_triangular_solve_attensor_attensor_bool_bool_bool(void* self, void* A, void* upper, void* transpose, void* unitriangular)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().triangular_solve(
        ((LanternObject<at::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get(), ((LanternObject<bool>*)transpose)->get(), ((LanternObject<bool>*)unitriangular)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_out_attensor_attensor_attensor_bool_bool(void* e, void* V, void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig_out(
        ((LanternObject<at::Tensor>*)e)->get(), ((LanternObject<at::Tensor>*)V)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_symeig_attensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::symeig(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_symeig_attensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().symeig(
        ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__symeig_helper_attensor_bool_bool(void* self, void* eigenvectors, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_symeig_helper(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get(), ((LanternObject<bool>*)upper)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_out_attensor_attensor_attensor_bool(void* e, void* v, void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig_out(
        ((LanternObject<at::Tensor>*)e)->get(), ((LanternObject<at::Tensor>*)v)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_eig_attensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::eig(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_eig_attensor_bool(void* self, void* eigenvectors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().eig(
        ((LanternObject<bool>*)eigenvectors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_out_attensor_attensor_attensor_attensor_bool_bool(void* U, void* S, void* V, void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd_out(
        ((LanternObject<at::Tensor>*)U)->get(), ((LanternObject<at::Tensor>*)S)->get(), ((LanternObject<at::Tensor>*)V)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_svd_attensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::svd(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_svd_attensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().svd(
        ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__svd_helper_attensor_bool_bool(void* self, void* some, void* compute_uv)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_svd_helper(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get(), ((LanternObject<bool>*)compute_uv)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_swapaxes_attensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::swapaxes(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapaxes_attensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().swapaxes(
        ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapaxes__attensor_intt_intt(void* self, void* axis0, void* axis1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().swapaxes_(
        ((LanternObject<int64_t>*)axis0)->get(), ((LanternObject<int64_t>*)axis1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_swapdims_attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::swapdims(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapdims_attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().swapdims(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_swapdims__attensor_intt_intt(void* self, void* dim0, void* dim1)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().swapdims_(
        ((LanternObject<int64_t>*)dim0)->get(), ((LanternObject<int64_t>*)dim1)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_out_attensor_attensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_attensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_attensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cholesky(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_out_attensor_attensor_attensor_bool(void* out, void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky_solve_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_solve_attensor_attensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky_solve(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_solve_attensor_attensor_bool(void* self, void* input2, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cholesky_solve(
        ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cholesky_solve_helper_attensor_attensor_bool(void* self, void* A, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cholesky_solve_helper(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_attensor_attensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_solve_attensor_attensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().solve(
        ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_solve_out_attensor_attensor_attensor_attensor(void* solution, void* lu, void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::solve_out(
        ((LanternObject<at::Tensor>*)solution)->get(), ((LanternObject<at::Tensor>*)lu)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__solve_helper_attensor_attensor(void* self, void* A)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_solve_helper(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)A)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_attensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky_inverse(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_cholesky_inverse_attensor_bool(void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().cholesky_inverse(
        ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_cholesky_inverse_out_attensor_attensor_bool(void* out, void* self, void* upper)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::cholesky_inverse_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)upper)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_out_attensor_attensor_attensor_bool(void* Q, void* R, void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr_out(
        ((LanternObject<at::Tensor>*)Q)->get(), ((LanternObject<at::Tensor>*)R)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_qr_attensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::qr(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_qr_attensor_bool(void* self, void* some)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().qr(
        ((LanternObject<bool>*)some)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_out_attensor_attensor_attensor(void* a, void* tau, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf_out(
        ((LanternObject<at::Tensor>*)a)->get(), ((LanternObject<at::Tensor>*)tau)->get(), ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_geqrf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::geqrf(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_geqrf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().geqrf(
        )));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_attensor_attensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::orgqr(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_orgqr_attensor_attensor(void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().orgqr(
        ((LanternObject<at::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_orgqr_out_attensor_attensor_attensor(void* out, void* self, void* input2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::orgqr_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_out_attensor_attensor_attensor_attensor_bool_bool(void* out, void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ormqr_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ormqr_attensor_attensor_attensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ormqr(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ormqr_attensor_attensor_attensor_bool_bool(void* self, void* input2, void* input3, void* left, void* transpose)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ormqr(
        ((LanternObject<at::Tensor>*)input2)->get(), ((LanternObject<at::Tensor>*)input3)->get(), ((LanternObject<bool>*)left)->get(), ((LanternObject<bool>*)transpose)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__lu_with_info_attensor_bool_bool(void* self, void* pivot, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_lu_with_info(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)pivot)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_out_attensor_attensor_attensor_attensor(void* out, void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lu_solve_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)LU_data)->get(), ((LanternObject<at::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_solve_attensor_attensor_attensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lu_solve(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)LU_data)->get(), ((LanternObject<at::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lu_solve_attensor_attensor_attensor(void* self, void* LU_data, void* LU_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lu_solve(
        ((LanternObject<at::Tensor>*)LU_data)->get(), ((LanternObject<at::Tensor>*)LU_pivots)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_unpack_attensor_attensor_bool_bool(void* LU_data, void* LU_pivots, void* unpack_data, void* unpack_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lu_unpack(
        ((LanternObject<at::Tensor>*)LU_data)->get(), ((LanternObject<at::Tensor>*)LU_pivots)->get(), ((LanternObject<bool>*)unpack_data)->get(), ((LanternObject<bool>*)unpack_pivots)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_lu_unpack_out_attensor_attensor_attensor_attensor_attensor_bool_bool(void* P, void* L, void* U, void* LU_data, void* LU_pivots, void* unpack_data, void* unpack_pivots)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::lu_unpack_out(
        ((LanternObject<at::Tensor>*)P)->get(), ((LanternObject<at::Tensor>*)L)->get(), ((LanternObject<at::Tensor>*)U)->get(), ((LanternObject<at::Tensor>*)LU_data)->get(), ((LanternObject<at::Tensor>*)LU_pivots)->get(), ((LanternObject<bool>*)unpack_data)->get(), ((LanternObject<bool>*)unpack_pivots)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_out_attensor_attensor_intt_bool_atgenerator(void* out, void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multinomial_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multinomial_attensor_intt_bool_atgenerator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multinomial(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_multinomial_attensor_intt_bool_atgenerator(void* self, void* num_samples, void* replacement, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().multinomial(
        ((LanternObject<int64_t>*)num_samples)->get(), ((LanternObject<bool>*)replacement)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lgamma_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lgamma_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_lgamma_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lgamma(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lgamma_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lgamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::digamma_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_digamma_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::digamma(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_digamma_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().digamma(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_polygamma_out_attensor_intt_attensor(void* out, void* n, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::polygamma_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_polygamma__attensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().polygamma_(
        ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erfinv(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erfinv(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_erfinv__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().erfinv_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_erfinv_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::erfinv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::i0(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().i0(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::i0_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_i0__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().i0_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_i0_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::i0_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sign(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sign(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sign__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().sign_(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_sign_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sign_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::signbit(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_signbit_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().signbit(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_signbit_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::signbit_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_dist_attensor_attensor_constatscalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::dist(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_dist_attensor_attensor_constatscalar(void* self, void* other, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().dist(
        ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atan2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atan2_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_atan2_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::atan2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_atan2_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().atan2(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_attensor_attensor_attensor_constatscalar(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lerp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<const at::Scalar &>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_out_attensor_attensor_attensor_attensor(void* out, void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lerp_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_attensor_attensor_constatscalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lerp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<const at::Scalar &>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_attensor_attensor_constatscalar(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lerp(
        ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<const at::Scalar &>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_lerp_attensor_attensor_attensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::lerp(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_lerp_attensor_attensor_attensor(void* self, void* end, void* weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().lerp(
        ((LanternObject<at::Tensor>*)end)->get(), ((LanternObject<at::Tensor>*)weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_out_attensor_attensor_intt_constatscalar_constatscalar(void* out, void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::histc_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<const at::Scalar &>*)min)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_histc_attensor_intt_constatscalar_constatscalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::histc(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<const at::Scalar &>*)min)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_histc_attensor_intt_constatscalar_constatscalar(void* self, void* bins, void* min, void* max)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().histc(
        ((LanternObject<int64_t>*)bins)->get(), ((LanternObject<const at::Scalar &>*)min)->get(), ((LanternObject<const at::Scalar &>*)max)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmod(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmod_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmod_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmod(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hypot_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hypot_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hypot(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().hypot(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_hypot__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().hypot_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_igamma_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::igamma_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_igamma_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::igamma(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igamma_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().igamma(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igamma__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().igamma_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_igammac_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::igammac_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_igammac_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::igammac(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igammac_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().igammac(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_igammac__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().igammac_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nextafter_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nextafter_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nextafter(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nextafter(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nextafter__attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nextafter_(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::remainder_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::remainder(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().remainder(
        ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::remainder_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_remainder_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::remainder(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_remainder_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().remainder(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::min(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().min(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fmin_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmin(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmin_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmin(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmin_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmin_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().max(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_fmax_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmax(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_fmax_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().fmax(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fmax_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fmax_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::maximum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_maximum_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().maximum(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_maximum_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::maximum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_max_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().max(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::minimum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_minimum_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().minimum(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_minimum_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::minimum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::min_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_min_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::min(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_min_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().min(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_attensor_attensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_attensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_attensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().quantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_attensor_attensor_attensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_attensor_attensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_attensor_attensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().quantile(
        ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_attensor_attensor_double_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_attensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_attensor_double_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nanquantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_attensor_attensor_attensor_intt_bool(void* out, void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_attensor_attensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_attensor_attensor_intt_bool(void* self, void* q, void* dim, void* keepdim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nanquantile(
        ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_attensor_attensor_double_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_attensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_attensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().quantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_out_attensor_attensor_attensor_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_quantile_attensor_attensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::quantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_quantile_attensor_attensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().quantile(
        ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_attensor_attensor_double_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_attensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_attensor_double_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nanquantile(
        ((LanternObject<double>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_out_attensor_attensor_attensor_intt_bool_stdstring(void* out, void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nanquantile_attensor_attensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nanquantile(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_nanquantile_attensor_attensor_intt_bool_stdstring(void* self, void* q, void* dim, void* keepdim, void* interpolation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().nanquantile(
        ((LanternObject<at::Tensor>*)q)->get(), ((LanternObject<c10::optional<int64_t>>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<std::string>*)interpolation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_attensor_attensor_attensor_intt_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_attensor_attensor_attensor_bool_intt_bool(void* values, void* indices, void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_attensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_attensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().sort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_attensor_bool_intt_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_attensor_bool_intt_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().sort(
        ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_attensor_attensor_attensor_atdimname_bool(void* values, void* indices, void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_out_attensor_attensor_attensor_bool_atdimname_bool(void* values, void* indices, void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_attensor_atdimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_attensor_atdimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().sort(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_sort_attensor_bool_atdimname_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::sort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_sort_attensor_bool_atdimname_bool(void* self, void* stable, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().sort(
        ((LanternObject<c10::optional<bool>>*)optional<bool>(stable).get())->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_msort_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::msort_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_msort_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::msort(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_msort_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().msort(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_attensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argsort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_attensor_intt_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().argsort(
        ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_argsort_attensor_atdimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::argsort(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_argsort_attensor_atdimname_bool(void* self, void* dim, void* descending)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().argsort(
        ((LanternObject<at::Dimname>*)dim)->get(), ((LanternObject<bool>*)descending)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_out_attensor_attensor_attensor_intt_intt_bool_bool(void* values, void* indices, void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk_out(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_topk_attensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::topk(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_topk_attensor_intt_intt_bool_bool(void* self, void* k, void* dim, void* largest, void* sorted)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(((LanternObject<at::Tensor>*)self)->get().topk(
        ((LanternObject<int64_t>*)k)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<bool>*)largest)->get(), ((LanternObject<bool>*)sorted)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_all_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::all(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_all_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().all(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_any_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::any(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_any_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().any(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_out_attensor_attensor_constatscalar_intt_constatscalar(void* out, void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::renorm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<const at::Scalar &>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_renorm_attensor_constatscalar_intt_constatscalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::renorm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<const at::Scalar &>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_renorm_attensor_constatscalar_intt_constatscalar(void* self, void* p, void* dim, void* maxnorm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().renorm(
        ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<const at::Scalar &>*)maxnorm)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_unfold_attensor_intt_intt_intt(void* self, void* dimension, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().unfold(
        ((LanternObject<int64_t>*)dimension)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unfold_backward_attensor_atintarrayref_intt_intt_intt(void* grad_in, void* input_sizes, void* dim, void* size, void* step)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::unfold_backward(
        ((LanternObject<at::Tensor>*)grad_in)->get(), ((LanternObject<at::IntArrayRef>*)input_sizes)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<int64_t>*)size)->get(), ((LanternObject<int64_t>*)step)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(torch::equal(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_equal_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<bool>(((LanternObject<at::Tensor>*)self)->get().equal(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_attensor_attensor_attensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pow(
        ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_attensor_constatscalar_attensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_constatscalar_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow(
        ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_out_attensor_attensor_constatscalar(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pow_attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pow(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow_attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pow(
        ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pow_(
        ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_pow__attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().pow_(
        ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_attensor_attensor_attensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power_attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().float_power(
        ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_attensor_constatscalar_attensor(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_constatscalar_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power(
        ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_out_attensor_attensor_constatscalar(void* out, void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_float_power_attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::float_power(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power_attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().float_power(
        ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power__attensor_constatscalar(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().float_power_(
        ((LanternObject<const at::Scalar &>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_float_power__attensor_attensor(void* self, void* exponent)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().float_power_(
        ((LanternObject<at::Tensor>*)exponent)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_normal__attensor_double_double_atgenerator(void* self, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().normal_(
        ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_attensor_attensor_double_atgenerator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::normal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_attensor_double_attensor_atgenerator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::normal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<double>*)mean)->get(), ((LanternObject<at::Tensor>*)std)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_attensor_attensor_attensor_atgenerator(void* out, void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::normal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)mean)->get(), ((LanternObject<at::Tensor>*)std)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_normal_out_attensor_double_double_atintarrayref_atgenerator(void* out, void* mean, void* std, void* size, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::normal_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<double>*)mean)->get(), ((LanternObject<double>*)std)->get(), ((LanternObject<at::IntArrayRef>*)size)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_alias_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::alias(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_alias_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().alias(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern__index_copy__attensor_intt_attensor_attensor(void* self, void* dim, void* index, void* source)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_index_copy_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<at::Tensor>*)index)->get(), ((LanternObject<at::Tensor>*)source)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cumsum(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumsum_out_attensor_attensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cumsum_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cumprod(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cumprod_out_attensor_attensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cumprod_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__amp_foreach_non_finite_check_and_unscale__attensorlist_attensor_attensor(void* self, void* found_inf, void* inv_scale)
{
  LANTERN_FUNCTION_START
    torch::_amp_foreach_non_finite_check_and_unscale_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::Tensor>*)found_inf)->get(), ((LanternObject<at::Tensor>*)inv_scale)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__amp_update_scale__attensor_attensor_attensor_double_double_intt(void* self, void* growth_tracker, void* found_inf, void* scale_growth_factor, void* scale_backoff_factor, void* growth_interval)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_amp_update_scale_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)growth_tracker)->get(), ((LanternObject<at::Tensor>*)found_inf)->get(), ((LanternObject<double>*)scale_growth_factor)->get(), ((LanternObject<double>*)scale_backoff_factor)->get(), ((LanternObject<int64_t>*)growth_interval)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_attensorlist_intt(void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cat(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__cat_out_attensor_attensorlist_intt(void* out, void* tensors, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_cat_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_attensorlist_constatscalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_add(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__attensorlist_constatscalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_attensorlist_constatscalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sub(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__attensorlist_constatscalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_attensorlist_constatscalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_mul(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__attensorlist_constatscalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_attensorlist_constatscalar(void* tensors, void* scalar)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_div(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__attensorlist_constatscalar(void* self, void* scalar)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<const at::Scalar &>*)scalar)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_attensorlist_attensorlist_constatscalar(void* tensors1, void* tensors2, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_add(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__attensorlist_attensorlist_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_attensorlist_attensorlist_constatscalar(void* tensors1, void* tensors2, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sub(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__attensorlist_attensorlist_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_attensorlist_attensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_mul(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__attensorlist_attensorlist(void* self, void* other)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)other)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_attensorlist_attensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_div(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__attensorlist_attensorlist(void* self, void* other)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)other)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add_attensorlist_atarrayrefatscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_add(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_add__attensorlist_atarrayrefatscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_add_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub_attensorlist_atarrayrefatscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sub(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sub__attensorlist_atarrayrefatscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sub_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div_attensorlist_atarrayrefatscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_div(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_div__attensorlist_atarrayrefatscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_div_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul_attensorlist_atarrayrefatscalar(void* tensors, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_mul(
        ((LanternObject<at::TensorList>*)tensors)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_mul__attensorlist_atarrayrefatscalar(void* self, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_mul_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_exp_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_exp(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_zero__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_zero_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_exp__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_exp_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sqrt_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sqrt(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sqrt__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sqrt_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_abs_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_abs(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_abs__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_abs_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_acos_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_acos(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_acos__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_acos_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_asin_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_asin(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_asin__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_asin_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_atan_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_atan(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_atan__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_atan_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_ceil_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_ceil(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_ceil__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_ceil_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cos_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_cos(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cos__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_cos_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cosh_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_cosh(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_cosh__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_cosh_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erf_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_erf(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erf__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_erf_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erfc_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_erfc(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_erfc__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_erfc_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_expm1_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_expm1(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_expm1__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_expm1_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_floor_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_floor(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_floor__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_floor_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_log(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log10_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_log10(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log10__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log10_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log1p_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_log1p(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log1p__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log1p_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log2_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_log2(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_log2__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_log2_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_neg_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_neg(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_neg__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_neg_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tan_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_tan(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tan__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_tan_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tanh_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_tanh(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_tanh__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_tanh_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sin_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sin(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sin__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sin_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sinh_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sinh(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sinh__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sinh_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_round_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_round(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_round__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_round_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_lgamma_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_lgamma(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_lgamma__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_lgamma_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_frac_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_frac(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_frac__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_frac_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_reciprocal_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_reciprocal(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_reciprocal__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_reciprocal_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sigmoid_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_sigmoid(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_sigmoid__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_sigmoid_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_trunc_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_trunc(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_trunc__attensorlist(void* self)
{
  LANTERN_FUNCTION_START
    torch::_foreach_trunc_(((LanternObject<at::TensorList>*)self)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv__attensorlist_attensorlist_attensorlist_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcdiv_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul__attensorlist_attensorlist_attensorlist_constatscalar(void* self, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcmul_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv__attensorlist_attensorlist_attensorlist_atarrayrefatscalar(void* self, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcdiv_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul__attensorlist_attensorlist_attensorlist_atarrayrefatscalar(void* self, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    torch::_foreach_addcmul_(((LanternObject<at::TensorList>*)self)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv_attensorlist_attensorlist_attensorlist_constatscalar(void* input, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_addcdiv(
        ((LanternObject<at::TensorList>*)input)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul_attensorlist_attensorlist_attensorlist_constatscalar(void* input, void* tensor1, void* tensor2, void* value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_addcmul(
        ((LanternObject<at::TensorList>*)input)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<const at::Scalar &>*)value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcdiv_attensorlist_attensorlist_attensorlist_atarrayrefatscalar(void* input, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_addcdiv(
        ((LanternObject<at::TensorList>*)input)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_addcmul_attensorlist_attensorlist_attensorlist_atarrayrefatscalar(void* input, void* tensor1, void* tensor2, void* scalars)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_addcmul(
        ((LanternObject<at::TensorList>*)input)->get(), ((LanternObject<at::TensorList>*)tensor1)->get(), ((LanternObject<at::TensorList>*)tensor2)->get(), ((LanternObject<at::ArrayRef<at::Scalar>>*)scalars)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_maximum_attensorlist_attensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_maximum(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__foreach_minimum_attensorlist_attensorlist(void* tensors1, void* tensors2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::_foreach_minimum(
        ((LanternObject<at::TensorList>*)tensors1)->get(), ((LanternObject<at::TensorList>*)tensors2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_attensor_attensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bucketize(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_out_attensor_attensor_attensor_bool_bool(void* out, void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bucketize_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_bucketize_constatscalar_attensor_bool_bool(void* self, void* boundaries, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::bucketize(
        ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)boundaries)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_attensor_attensor_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::searchsorted(
        ((LanternObject<at::Tensor>*)sorted_sequence)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_out_attensor_attensor_attensor_bool_bool(void* out, void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::searchsorted_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)sorted_sequence)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_searchsorted_attensor_constatscalar_bool_bool(void* sorted_sequence, void* self, void* out_int32, void* right)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::searchsorted(
        ((LanternObject<at::Tensor>*)sorted_sequence)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<bool>*)out_int32)->get(), ((LanternObject<bool>*)right)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_out_attensor_attensor_attensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mse_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_attensor_attensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mse_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_out_attensor_attensor_attensor_attensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mse_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mse_loss_backward_attensor_attensor_attensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mse_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_out_attensor_attensor_attensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::l1_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_attensor_attensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::l1_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_out_attensor_attensor_attensor_attensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::l1_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_l1_loss_backward_attensor_attensor_attensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::l1_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_out_attensor_attensor_attensor_constatscalar_constatscalar_attensor_intt(void* out, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multi_margin_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<const at::Scalar &>*)margin)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_attensor_attensor_constatscalar_constatscalar_attensor_intt(void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multi_margin_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<const at::Scalar &>*)margin)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_out_attensor_attensor_attensor_attensor_constatscalar_constatscalar_attensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multi_margin_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<const at::Scalar &>*)margin)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multi_margin_loss_backward_attensor_attensor_attensor_constatscalar_constatscalar_attensor_intt(void* grad_output, void* self, void* target, void* p, void* margin, void* weight, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multi_margin_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<const at::Scalar &>*)p)->get(), ((LanternObject<const at::Scalar &>*)margin)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_out_attensor_attensor_attensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multilabel_margin_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_attensor_attensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multilabel_margin_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_out_attensor_attensor_attensor_attensor_intt(void* output, void* is_target, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)is_target)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_forward_attensor_attensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::multilabel_margin_loss_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_out_attensor_attensor_attensor_attensor_intt_attensor(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multilabel_margin_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<at::Tensor>*)is_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_multilabel_margin_loss_backward_attensor_attensor_attensor_intt_attensor(void* grad_output, void* self, void* target, void* reduction, void* is_target)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::multilabel_margin_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<at::Tensor>*)is_target)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_out_attensor_attensor_attensor_attensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_nd_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss_nd(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_out_attensor_attensor_attensor_attensor_attensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)total_weight)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_forward_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_out_attensor_attensor_attensor_attensor_attensor_intt_intt_attensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<at::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss_backward_attensor_attensor_attensor_attensor_intt_intt_attensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<at::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_out_attensor_attensor_attensor_attensor_intt_intt(void* out, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_out_attensor_attensor_attensor_attensor_attensor_intt_intt(void* output, void* total_weight, void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)total_weight)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_forward_attensor_attensor_attensor_intt_intt(void* self, void* target, void* weight, void* reduction, void* ignore_index)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::nll_loss2d_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_out_attensor_attensor_attensor_attensor_attensor_intt_intt_attensor(void* grad_input, void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<at::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_nll_loss2d_backward_attensor_attensor_attensor_attensor_intt_intt_attensor(void* grad_output, void* self, void* target, void* weight, void* reduction, void* ignore_index, void* total_weight)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::nll_loss2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(weight).get())->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<int64_t>*)ignore_index)->get(), ((LanternObject<at::Tensor>*)total_weight)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_out_attensor_attensor_attensor_intt_double(void* out, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::smooth_l1_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_attensor_attensor_intt_double(void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::smooth_l1_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_out_attensor_attensor_attensor_attensor_intt_double(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::smooth_l1_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_smooth_l1_loss_backward_attensor_attensor_attensor_intt_double(void* grad_output, void* self, void* target, void* reduction, void* beta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::smooth_l1_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)beta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_out_attensor_attensor_attensor_intt_double(void* out, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::huber_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_attensor_attensor_intt_double(void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::huber_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_backward_out_attensor_attensor_attensor_attensor_intt_double(void* grad_input, void* grad_output, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::huber_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_huber_loss_backward_attensor_attensor_attensor_intt_double(void* grad_output, void* self, void* target, void* reduction, void* delta)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::huber_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get(), ((LanternObject<double>*)delta)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_out_attensor_attensor_attensor_intt(void* out, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::soft_margin_loss_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_attensor_attensor_intt(void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::soft_margin_loss(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_out_attensor_attensor_attensor_attensor_intt(void* grad_input, void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::soft_margin_loss_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_soft_margin_loss_backward_attensor_attensor_attensor_intt(void* grad_output, void* self, void* target, void* reduction)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::soft_margin_loss_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)target)->get(), ((LanternObject<int64_t>*)reduction)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_out_attensor_attensor_constatscalar_constatscalar_constatscalar(void* out, void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::elu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get(), ((LanternObject<const at::Scalar &>*)scale)->get(), ((LanternObject<const at::Scalar &>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_attensor_constatscalar_constatscalar_constatscalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::elu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get(), ((LanternObject<const at::Scalar &>*)scale)->get(), ((LanternObject<const at::Scalar &>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu_backward_attensor_constatscalar_constatscalar_constatscalar_bool_attensor(void* grad_output, void* alpha, void* scale, void* input_scale, void* is_result, void* self_or_result)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::elu_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get(), ((LanternObject<const at::Scalar &>*)scale)->get(), ((LanternObject<const at::Scalar &>*)input_scale)->get(), ((LanternObject<bool>*)is_result)->get(), ((LanternObject<at::Tensor>*)self_or_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_elu__attensor_constatscalar_constatscalar_constatscalar(void* self, void* alpha, void* scale, void* input_scale)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::elu_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get(), ((LanternObject<const at::Scalar &>*)scale)->get(), ((LanternObject<const at::Scalar &>*)input_scale)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_out_attensor_attensor_intt(void* out, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::glu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_attensor_intt(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::glu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_out_attensor_attensor_attensor_intt(void* grad_input, void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::glu_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_glu_backward_attensor_attensor_intt(void* grad_output, void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::glu_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardsigmoid_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardsigmoid(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardsigmoid_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardsigmoid_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardsigmoid_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_out_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardtanh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min_val)->get(), ((LanternObject<const at::Scalar &>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_attensor_constatscalar_constatscalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardtanh(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min_val)->get(), ((LanternObject<const at::Scalar &>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_out_attensor_attensor_attensor_constatscalar_constatscalar(void* grad_input, void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardtanh_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min_val)->get(), ((LanternObject<const at::Scalar &>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh_backward_attensor_attensor_constatscalar_constatscalar(void* grad_output, void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardtanh_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min_val)->get(), ((LanternObject<const at::Scalar &>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardtanh__attensor_constatscalar_constatscalar(void* self, void* min_val, void* max_val)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardtanh_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)min_val)->get(), ((LanternObject<const at::Scalar &>*)max_val)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardswish_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardswish(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish__attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardswish_(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_hardswish_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::hardswish_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_out_attensor_attensor_constatscalar(void* out, void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::leaky_relu_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_attensor_constatscalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::leaky_relu(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu_backward_attensor_attensor_constatscalar_bool(void* grad_output, void* self, void* negative_slope, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::leaky_relu_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)negative_slope)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_leaky_relu__attensor_constatscalar(void* self, void* negative_slope)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::leaky_relu_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)negative_slope)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_sigmoid_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_sigmoid(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_out_attensor_attensor_attensor(void* output, void* buffer, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)buffer)->get(), ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_forward_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::log_sigmoid_forward(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_out_attensor_attensor_attensor_attensor(void* grad_input, void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_sigmoid_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)buffer)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_log_sigmoid_backward_attensor_attensor_attensor(void* grad_output, void* self, void* buffer)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::log_sigmoid_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)buffer)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_out_attensor_attensor_attensor_constatscalar_constatscalar_bool_atgenerator(void* out, void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu_with_noise_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)noise)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_attensor_attensor_constatscalar_constatscalar_bool_atgenerator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu_with_noise(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)noise)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise_backward_attensor_attensor_attensor_constatscalar_constatscalar_bool_bool(void* grad_output, void* self, void* noise, void* lower, void* upper, void* training, void* self_is_result)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu_with_noise_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)noise)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<bool>*)self_is_result)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_rrelu_with_noise__attensor_attensor_constatscalar_constatscalar_bool_atgenerator(void* self, void* noise, void* lower, void* upper, void* training, void* generator)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::rrelu_with_noise_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)noise)->get(), ((LanternObject<const at::Scalar &>*)lower)->get(), ((LanternObject<const at::Scalar &>*)upper)->get(), ((LanternObject<bool>*)training)->get(), ((LanternObject<c10::optional<at::Generator>>*)optional<at::Generator>(generator).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_out_attensor_attensor_constatscalar_constatscalar(void* out, void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softplus_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_attensor_constatscalar_constatscalar(void* self, void* beta, void* threshold)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softplus(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_out_attensor_attensor_attensor_constatscalar_constatscalar_attensor(void* grad_input, void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softplus_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softplus_backward_attensor_attensor_constatscalar_constatscalar_attensor(void* grad_output, void* self, void* beta, void* threshold, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softplus_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)beta)->get(), ((LanternObject<const at::Scalar &>*)threshold)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_out_attensor_attensor_constatscalar(void* out, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softshrink_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_attensor_constatscalar(void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softshrink(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_out_attensor_attensor_attensor_constatscalar(void* grad_input, void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softshrink_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_softshrink_backward_attensor_attensor_constatscalar(void* grad_output, void* self, void* lambd)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::softshrink_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)lambd)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_out_attensor_attensor_atintarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool2d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_adaptive_avg_pool2d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_adaptive_avg_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_mkldnn_adaptive_avg_pool2d_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::mkldnn_adaptive_avg_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_adaptive_avg_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool2d_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_adaptive_avg_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_out_attensor_attensor_atintarrayref(void* out, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool3d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_adaptive_avg_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_avg_pool3d_backward_out_attensor_attensor_attensor(void* grad_input, void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_avg_pool3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__adaptive_avg_pool3d_backward_attensor_attensor(void* grad_output, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_adaptive_avg_pool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_out_attensor_attensor_attensor_atintarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_out_attensor_attensor_attensor_attensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_max_pool2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool2d_backward_attensor_attensor_attensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_max_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_out_attensor_attensor_attensor_atintarrayref(void* out, void* indices, void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_attensor_atintarrayref(void* self, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::adaptive_max_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_out_attensor_attensor_attensor_attensor(void* grad_input, void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_max_pool3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_adaptive_max_pool3d_backward_attensor_attensor_attensor(void* grad_output, void* self, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::adaptive_max_pool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool2d_backward_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* out, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_avg_pool3d_backward_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_bool_bool_intt(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* ceil_mode, void* count_include_pad, void* divisor_override)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::avg_pool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<bool>*)count_include_pad)->get(), ((LanternObject<c10::optional<int64_t>>*)divisor_override)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_out_attensor_attensor_attensor_atintarrayref_atintarrayref_attensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_attensor_atintarrayref_atintarrayref_attensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_attensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fractional_max_pool2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool2d_backward_attensor_attensor_atintarrayref_atintarrayref_attensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fractional_max_pool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_out_attensor_attensor_attensor_atintarrayref_atintarrayref_attensor(void* output, void* indices, void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_attensor_atintarrayref_atintarrayref_attensor(void* self, void* kernel_size, void* output_size, void* random_samples)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::fractional_max_pool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)random_samples)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_attensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fractional_max_pool3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fractional_max_pool3d_backward_attensor_attensor_atintarrayref_atintarrayref_attensor(void* grad_output, void* self, void* kernel_size, void* output_size, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fractional_max_pool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool2d_with_indices(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool_attensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool2d_with_indices_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool2d_with_indices_backward_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool_attensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool2d_with_indices_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* out, void* indices, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool(void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::max_pool3d_with_indices(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool_attensor(void* grad_input, void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool3d_with_indices_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_pool3d_with_indices_backward_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_bool_attensor(void* grad_output, void* self, void* kernel_size, void* stride, void* padding, void* dilation, void* ceil_mode, void* indices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_pool3d_with_indices_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<bool>*)ceil_mode)->get(), ((LanternObject<at::Tensor>*)indices)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_out_attensor_attensor_attensor_atintarrayref(void* out, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_attensor_attensor_atintarrayref(void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_out_attensor_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool2d_backward_attensor_attensor_attensor_atintarrayref(void* grad_output, void* self, void* indices, void* output_size)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_out_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_attensor_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_out_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref(void* grad_input, void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_max_unpool3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref(void* grad_output, void* self, void* indices, void* output_size, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::max_unpool3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)indices)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_out_attensor_attensor_atintarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad1d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_attensor_atintarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_out_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad1d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad1d_backward_attensor_attensor_atintarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_out_attensor_attensor_atintarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_attensor_atintarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_out_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_reflection_pad2d_backward_attensor_attensor_atintarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::reflection_pad2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_out_attensor_attensor_atintarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad1d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_attensor_atintarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_out_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad1d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad1d_backward_attensor_attensor_atintarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_out_attensor_attensor_atintarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_attensor_atintarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_out_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad2d_backward_attensor_attensor_atintarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_out_attensor_attensor_atintarrayref(void* out, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_attensor_atintarrayref(void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_out_attensor_attensor_attensor_atintarrayref(void* grad_input, void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_replication_pad3d_backward_attensor_attensor_atintarrayref(void* grad_output, void* self, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::replication_pad3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_attensor_atintarrayref_bool_atarrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_attensor_atintarrayref_atintarrayref_bool_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_attensor_atintarrayref_bool_atarrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_attensor_atintarrayref_atintarrayref_bool_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_attensor_atintarrayref_bool_atarrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_attensor_atintarrayref_atintarrayref_bool_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_attensor_atintarrayref_bool_atarrayrefdouble(void* input, void* output_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_attensor_atintarrayref_atintarrayref_bool_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_attensor_atintarrayref_atarrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_attensor_atintarrayref_atintarrayref_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_attensor_atintarrayref_atarrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_attensor_atintarrayref_atintarrayref_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_attensor_atintarrayref_atarrayrefdouble(void* input, void* output_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_attensor_atintarrayref_atintarrayref_atarrayrefdouble(void* grad_output, void* output_size, void* input_size, void* scale_factors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(output_size).get())->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(scale_factors).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_out_attensor_attensor_atintarrayref_bool_double(void* out, void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_attensor_atintarrayref_bool_double(void* self, void* output_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_out_attensor_attensor_atintarrayref_atintarrayref_bool_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_linear1d_backward_attensor_atintarrayref_atintarrayref_bool_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_linear1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_out_attensor_attensor_atintarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_attensor_atintarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_out_attensor_attensor_atintarrayref_atintarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bilinear2d_backward_attensor_atintarrayref_atintarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bilinear2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_out_attensor_attensor_atintarrayref_bool_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_attensor_atintarrayref_bool_double_double(void* self, void* output_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_out_attensor_attensor_atintarrayref_atintarrayref_bool_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_bicubic2d_backward_attensor_atintarrayref_atintarrayref_bool_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_bicubic2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_out_attensor_attensor_atintarrayref_bool_double_double_double(void* out, void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_attensor_atintarrayref_bool_double_double_double(void* self, void* output_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_out_attensor_attensor_atintarrayref_atintarrayref_bool_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_trilinear3d_backward_attensor_atintarrayref_atintarrayref_bool_double_double_double(void* grad_output, void* output_size, void* input_size, void* align_corners, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_trilinear3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<bool>*)align_corners)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_out_attensor_attensor_atintarrayref_double(void* out, void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_attensor_atintarrayref_double(void* self, void* output_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_out_attensor_attensor_atintarrayref_atintarrayref_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest1d_backward_attensor_atintarrayref_atintarrayref_double(void* grad_output, void* output_size, void* input_size, void* scales)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest1d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_out_attensor_attensor_atintarrayref_double_double(void* out, void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_attensor_atintarrayref_double_double(void* self, void* output_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_out_attensor_attensor_atintarrayref_atintarrayref_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest2d_backward_attensor_atintarrayref_atintarrayref_double_double(void* grad_output, void* output_size, void* input_size, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_out_attensor_attensor_atintarrayref_double_double_double(void* out, void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_attensor_atintarrayref_double_double_double(void* self, void* output_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_out_attensor_attensor_atintarrayref_atintarrayref_double_double_double(void* grad_input, void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_upsample_nearest3d_backward_attensor_atintarrayref_atintarrayref_double_double_double(void* grad_output, void* output_size, void* input_size, void* scales_d, void* scales_h, void* scales_w)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::upsample_nearest3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<c10::optional<double>>*)scales_d)->get(), ((LanternObject<c10::optional<double>>*)scales_h)->get(), ((LanternObject<c10::optional<double>>*)scales_w)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_out_attensor_attensor_attensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sigmoid_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_sigmoid_backward_attensor_attensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::sigmoid_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_out_attensor_attensor_attensor_double(void* grad_input, void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logit_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_logit_backward_attensor_attensor_double(void* grad_output, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::logit_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_out_attensor_attensor_attensor(void* grad_input, void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tanh_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_tanh_backward_attensor_attensor(void* grad_output, void* output)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::tanh_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)output)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_transpose2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_transpose2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_out_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref_attensor_attensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_bias)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::Tensor>*)columns)->get(), ((LanternObject<at::Tensor>*)ones)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose2d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref_attensor_attensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* columns, void* ones, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::Tensor>*)columns)->get(), ((LanternObject<at::Tensor>*)ones)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_transpose3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* output_padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_transpose3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_out_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref_attensor_attensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_bias)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_transpose3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref_attensor_attensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* output_padding, void* dilation, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_transpose3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)output_padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_out_attensor_attensor_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_forward_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_out_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_attensor_attensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_bias)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv2d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_attensor_attensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv_depthwise2d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv_depthwise2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv_depthwise2d_forward_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_forward_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::thnn_conv_depthwise2d_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_out_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_input, void* grad_weight, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_thnn_conv_depthwise2d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::thnn_conv_depthwise2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<std::array<bool,2>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::conv_depthwise3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_backward_out_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_depthwise3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_bias)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_conv_depthwise3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::conv_depthwise3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_out_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* out, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv3d_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_out_attensor_attensor_attensor_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* output, void* finput, void* fgrad_input, void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward_out(
        ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_forward_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_forward(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_out_attensor_attensor_attensor_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_attensor_attensor(void* grad_input, void* grad_weight, void* grad_bias, void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_weight)->get(), ((LanternObject<at::Tensor>*)grad_bias)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_attensor_attensor_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* finput, void* fgrad_input, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::Tensor>*)finput)->get(), ((LanternObject<at::Tensor>*)fgrad_input)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_dilated2d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated2d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated2d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_attensor_attensor_atintarrayref_attensor_atintarrayref_atintarrayref_atintarrayref(void* self, void* weight, void* kernel_size, void* bias, void* stride, void* padding, void* dilation)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::slow_conv_dilated3d(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(bias).get())->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_slow_conv_dilated3d_backward_attensor_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_stdarraybool(void* grad_output, void* self, void* weight, void* kernel_size, void* stride, void* padding, void* dilation, void* output_mask)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::slow_conv_dilated3d_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)weight)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<std::array<bool,3>>*)output_mask)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::col2im_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* self, void* output_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::col2im(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)output_size)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_input, void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::col2im_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_col2im_backward_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_output, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::col2im_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_column_stack_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::column_stack(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_column_stack_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::column_stack_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* out, void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::im2col_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* self, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::im2col(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_out_attensor_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_input, void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::im2col_backward_out(
        ((LanternObject<at::Tensor>*)grad_input)->get(), ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_im2col_backward_attensor_atintarrayref_atintarrayref_atintarrayref_atintarrayref_atintarrayref(void* grad_output, void* input_size, void* kernel_size, void* dilation, void* padding, void* stride)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::im2col_backward(
        ((LanternObject<at::Tensor>*)grad_output)->get(), ((LanternObject<at::IntArrayRef>*)input_size)->get(), ((LanternObject<at::IntArrayRef>*)kernel_size)->get(), ((LanternObject<at::IntArrayRef>*)dilation)->get(), ((LanternObject<at::IntArrayRef>*)padding)->get(), ((LanternObject<at::IntArrayRef>*)stride)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isfinite_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isfinite(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isfinite_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isfinite(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isinf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isinf(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isinf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_record_stream_attensor_atstream(void* self, void* s)
{
  LANTERN_FUNCTION_START
    ((LanternObject<at::Tensor>*)self)->get().record_stream(((LanternObject<at::Stream>*)s)->get());
    return NULL;
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isposinf(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isposinf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isposinf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isposinf_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isposinf_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isneginf(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_isneginf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().isneginf(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_isneginf_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::isneginf_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__add_batch_dim_attensor_intt_intt(void* self, void* batch_dim, void* level)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_add_batch_dim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)batch_dim)->get(), ((LanternObject<int64_t>*)level)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__remove_batch_dim_attensor_intt_intt_intt(void* self, void* level, void* batch_size, void* out_dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_remove_batch_dim(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)level)->get(), ((LanternObject<int64_t>*)batch_size)->get(), ((LanternObject<int64_t>*)out_dim)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_entr_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_entr(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_entr_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_entr_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expm1_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_expm1(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expm1_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_expm1_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_exp2_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_exp2(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_exp2_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_exp2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_gammaln_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_gammaln(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_gammaln_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_gammaln_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erf_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erf(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erf_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erf_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfc_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erfc(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfc_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erfc_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfinv_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erfinv(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_erfinv_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_erfinv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_constatscalar_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py(
        ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_attensor_constatscalar(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_attensor_constatscalar_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<const at::Scalar &>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_xlog1py_out_attensor_attensor_constatscalar(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_xlog1py_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_i0e_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_i0e(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_i0e_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_i0e_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_logit_attensor_double(void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_logit(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_logit_out_attensor_attensor_double(void* out, void* self, void* eps)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_logit_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)eps)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expit_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_expit(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_special_expit_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::special_expit_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_hfft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_hfft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_hfft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_hfft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ihfft_attensor_intt_intt_stdstring(void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ihfft(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ihfft_out_attensor_attensor_intt_intt_stdstring(void* out, void* self, void* n, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ihfft_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<int64_t>>*)n)->get(), ((LanternObject<int64_t>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft2_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fft2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fft2_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fft2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft2_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifft2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifft2_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifft2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft2_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfft2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfft2_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfft2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft2_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfft2(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfft2_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfft2_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftn_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fftn(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftn_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fftn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftn_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifftn(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftn_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifftn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftn_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfftn(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftn_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfftn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfftn_attensor_atintarrayref_atintarrayref_stdstring(void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfftn(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_irfftn_out_attensor_attensor_atintarrayref_atintarrayref_stdstring(void* out, void* self, void* s, void* dim, void* norm)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_irfftn_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(s).get())->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(norm).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftfreq_intt_double_attensoroptions(void* n, void* d, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fftfreq(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftfreq_out_attensor_intt_double(void* out, void* n, void* d)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fftfreq_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftfreq_intt_double_attensoroptions(void* n, void* d, void* options)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfftfreq(
        ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get(), ((LanternObject<at::TensorOptions>*)options)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_rfftfreq_out_attensor_intt_double(void* out, void* n, void* d)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_rfftfreq_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<int64_t>*)n)->get(), ((LanternObject<double>*)d)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_fftshift_attensor_atintarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_fftshift(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_fft_ifftshift_attensor_atintarrayref(void* self, void* dim)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::fft_ifftshift(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_ex_attensor_bool(void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_cholesky_ex(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_ex_out_attensor_attensor_attensor_bool(void* L, void* info, void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_cholesky_ex_out(
        ((LanternObject<at::Tensor>*)L)->get(), ((LanternObject<at::Tensor>*)info)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cholesky(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cholesky_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cholesky_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_det_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_det(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_det_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_det_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_det_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::det(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_det_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().det(
        ));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_lstsq_attensor_attensor_double_stdstring(void* self, void* b, void* rcond, void* driver)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_lstsq(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)b)->get(), ((LanternObject<c10::optional<double>>*)rcond)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(driver).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_lstsq_out_attensor_attensor_attensor_attensor_attensor_attensor_double_stdstring(void* solution, void* residuals, void* rank, void* singular_values, void* self, void* b, void* rcond, void* driver)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_lstsq_out(
        ((LanternObject<at::Tensor>*)solution)->get(), ((LanternObject<at::Tensor>*)residuals)->get(), ((LanternObject<at::Tensor>*)rank)->get(), ((LanternObject<at::Tensor>*)singular_values)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)b)->get(), ((LanternObject<c10::optional<double>>*)rcond)->get(), ((LanternObject<c10::optional<std::string>>*)optional<std::string>(driver).get())->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_slogdet_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_slogdet(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_slogdet_out_attensor_attensor_attensor(void* sign, void* logabsdet, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_slogdet_out(
        ((LanternObject<at::Tensor>*)sign)->get(), ((LanternObject<at::Tensor>*)logabsdet)->get(), ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eig_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eig(
        ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eig_out_attensor_attensor_attensor(void* eigenvalues, void* eigenvectors, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eig_out(
        ((LanternObject<at::Tensor>*)eigenvalues)->get(), ((LanternObject<at::Tensor>*)eigenvectors)->get(), ((LanternObject<at::Tensor>*)self)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvals_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_eigvals(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvals_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_eigvals_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigh_attensor_stdstring(void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eigh(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)UPLO)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigh_out_attensor_attensor_attensor_stdstring(void* eigvals, void* eigvecs, void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_eigh_out(
        ((LanternObject<at::Tensor>*)eigvals)->get(), ((LanternObject<at::Tensor>*)eigvecs)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)UPLO)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvalsh_attensor_stdstring(void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_eigvalsh(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)UPLO)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_eigvalsh_out_attensor_attensor_stdstring(void* out, void* self, void* UPLO)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_eigvalsh_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)UPLO)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_householder_product_attensor_attensor(void* input, void* tau)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_householder_product(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)tau)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_householder_product_out_attensor_attensor_attensor(void* out, void* input, void* tau)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_householder_product_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)tau)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_inv_out_helper__attensor_attensor_attensor(void* self, void* infos_lu, void* infos_getri)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_linalg_inv_out_helper_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)infos_lu)->get(), ((LanternObject<at::Tensor>*)infos_getri)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_ex_attensor_bool(void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_inv_ex(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_ex_out_attensor_attensor_attensor_bool(void* inverse, void* info, void* self, void* check_errors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_inv_ex_out(
        ((LanternObject<at::Tensor>*)inverse)->get(), ((LanternObject<at::Tensor>*)info)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)check_errors)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_attensor(void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_inv(
        ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_inv_out_attensor_attensor(void* out, void* self)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_inv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_inner_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::inner(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_inner_attensor_attensor(void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().inner(
        ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_inner_out_attensor_attensor_attensor(void* out, void* self, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::inner_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_attensor_attensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::outer(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_outer_attensor_attensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().outer(
        ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_outer_out_attensor_attensor_attensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::outer_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_attensor_attensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ger(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_Tensor_ger_attensor_attensor(void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(((LanternObject<at::Tensor>*)self)->get().ger(
        ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_ger_out_attensor_attensor_attensor(void* out, void* self, void* vec2)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::ger_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)vec2)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_attensor_stdstring_atintarrayref_bool_atscalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_attensor_attensor_constatscalar_atintarrayref_bool_atscalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_norm_out_attensor_attensor_stdstring_atintarrayref_bool_atscalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_vector_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_vector_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_vector_norm_out_attensor_attensor_constatscalar_atintarrayref_bool_atscalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_vector_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)ord)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dim).get())->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_attensor_constatscalar_atintarrayref_bool_atscalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)ord)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_out_attensor_attensor_constatscalar_atintarrayref_bool_atscalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<const at::Scalar &>*)ord)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_attensor_stdstring_atintarrayref_bool_atscalartype(void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_norm(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_norm_out_attensor_attensor_stdstring_atintarrayref_bool_atscalartype(void* out, void* self, void* ord, void* dim, void* keepdim, void* dtype)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_norm_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)ord)->get(), ((LanternObject<at::IntArrayRef>*)dim)->get(), ((LanternObject<bool>*)keepdim)->get(), ((LanternObject<c10::optional<at::ScalarType>>*)optional<at::ScalarType>(dtype).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svd_out_attensor_attensor_attensor_attensor_bool(void* U, void* S, void* Vh, void* self, void* full_matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_svd_out(
        ((LanternObject<at::Tensor>*)U)->get(), ((LanternObject<at::Tensor>*)S)->get(), ((LanternObject<at::Tensor>*)Vh)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)full_matrices)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svd_attensor_bool(void* self, void* full_matrices)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_svd(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<bool>*)full_matrices)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svdvals_attensor(void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_svdvals(
        ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_svdvals_out_attensor_attensor(void* out, void* input)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_svdvals_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_attensor_constatscalar(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cond(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_out_attensor_attensor_constatscalar(void* out, void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cond_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<at::Scalar>>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_attensor_stdstring(void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cond(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_cond_out_attensor_attensor_stdstring(void* out, void* self, void* p)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_cond_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)p)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_attensor_double_bool(void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_pinv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_attensor_attensor_bool(void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_pinv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_out_attensor_attensor_double_bool(void* out, void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_pinv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<double>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_pinv_out_attensor_attensor_attensor_bool(void* out, void* self, void* rcond, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_pinv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)rcond)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_solve_out_helper__attensor_attensor_attensor(void* self, void* other, void* infos)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_linalg_solve_out_helper_(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<at::Tensor>*)infos)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_solve_attensor_attensor(void* input, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_solve(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_solve_out_attensor_attensor_attensor(void* out, void* input, void* other)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_solve_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)other)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorinv_attensor_intt(void* self, void* ind)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_tensorinv(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)ind)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorinv_out_attensor_attensor_intt(void* out, void* self, void* ind)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_tensorinv_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)ind)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorsolve_attensor_attensor_atintarrayref(void* self, void* other, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_tensorsolve(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dims).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_tensorsolve_out_attensor_attensor_attensor_atintarrayref(void* out, void* self, void* other, void* dims)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_tensorsolve_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(dims).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_qr_attensor_stdstring(void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_qr(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_qr_out_attensor_attensor_attensor_stdstring(void* Q, void* R, void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::linalg_qr_out(
        ((LanternObject<at::Tensor>*)Q)->get(), ((LanternObject<at::Tensor>*)R)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern__linalg_qr_helper_attensor_stdstring(void* self, void* mode)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<std::vector<void*>>(to_vector(torch::_linalg_qr_helper(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<std::string>*)mode)->get())));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_power_attensor_intt(void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_power(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_power_out_attensor_attensor_intt(void* out, void* self, void* n)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_power_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<int64_t>*)n)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_attensor_double_bool(void* self, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_rank(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_out_attensor_attensor_double_bool(void* out, void* self, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_rank_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<c10::optional<double>>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_attensor_attensor_bool(void* input, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_rank(
        ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_matrix_rank_out_attensor_attensor_attensor_bool(void* out, void* input, void* tol, void* hermitian)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_matrix_rank_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::Tensor>*)input)->get(), ((LanternObject<at::Tensor>*)tol)->get(), ((LanternObject<bool>*)hermitian)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_multi_dot_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_multi_dot(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_linalg_multi_dot_out_attensor_attensorlist(void* out, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::linalg_multi_dot_out(
        ((LanternObject<at::Tensor>*)out)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_serialization_subcmul_attensor_attensor_constatscalar(void* self, void* other, void* alpha)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_serialization_subcmul(
        ((LanternObject<at::Tensor>*)self)->get(), ((LanternObject<at::Tensor>*)other)->get(), ((LanternObject<const at::Scalar &>*)alpha)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_intlist_attensor_atintarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_optional_intlist(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(addends).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_filled_intlist_attensor_atintarrayref(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_optional_filled_intlist(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<c10::optional<at::IntArrayRef>>*)optional<at::IntArrayRef>(addends).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_optional_floatlist_attensor_atarrayrefdouble(void* values, void* addends)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_optional_floatlist(
        ((LanternObject<at::Tensor>*)values)->get(), ((LanternObject<c10::optional<at::ArrayRef<double>>>*)optional<at::ArrayRef<double>>(addends).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_string_default_attensor_stdstring_stdstring(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_string_default(
        ((LanternObject<at::Tensor>*)dummy)->get(), ((LanternObject<std::string>*)a)->get(), ((LanternObject<std::string>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_ambiguous_defaults_attensor_intt_intt(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_ambiguous_defaults(
        ((LanternObject<at::Tensor>*)dummy)->get(), ((LanternObject<int64_t>*)a)->get(), ((LanternObject<int64_t>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern__test_ambiguous_defaults_attensor_intt_stdstring(void* dummy, void* a, void* b)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::_test_ambiguous_defaults(
        ((LanternObject<at::Tensor>*)dummy)->get(), ((LanternObject<int64_t>*)a)->get(), ((LanternObject<std::string>*)b)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_segment_reduce_attensor_stdstring_attensor_attensor_intt_bool_constatscalar(void* data, void* reduce, void* lengths, void* indices, void* axis, void* unsafe, void* initial)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::segment_reduce(
        ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<std::string>*)reduce)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(lengths).get())->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(indices).get())->get(), ((LanternObject<int64_t>*)axis)->get(), ((LanternObject<bool>*)unsafe)->get(), ((LanternObject<c10::optional<at::Scalar>>*)initial)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_segment_reduce_backward_attensor_attensor_attensor_attensor(void* grad, void* output, void* data, void* lengths)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::segment_reduce_backward(
        ((LanternObject<at::Tensor>*)grad)->get(), ((LanternObject<at::Tensor>*)output)->get(), ((LanternObject<at::Tensor>*)data)->get(), ((LanternObject<c10::optional<at::Tensor>>*)optional<at::Tensor>(lengths).get())->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_pad_sequence_attensorlist_bool_double(void* sequences, void* batch_first, void* padding_value)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::pad_sequence(
        ((LanternObject<at::TensorList>*)sequences)->get(), ((LanternObject<bool>*)batch_first)->get(), ((LanternObject<double>*)padding_value)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_flatten_dense_tensors_attensorlist(void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::Tensor>(torch::flatten_dense_tensors(
        ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

void* _lantern_unflatten_dense_tensors_attensor_attensorlist(void* flat, void* tensors)
{
  LANTERN_FUNCTION_START
    return (void *) new LanternObject<at::TensorList>(torch::unflatten_dense_tensors(
        ((LanternObject<at::Tensor>*)flat)->get(), ((LanternObject<at::TensorList>*)tensors)->get()));
  LANTERN_FUNCTION_END
}

/* Autogen Body -- End */
