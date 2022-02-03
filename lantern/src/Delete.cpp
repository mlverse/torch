#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "Function.h"
#include "lantern/lantern.h"
#include "utils.hpp"

template <class T>
void lantern_delete(void *x) {
  delete reinterpret_cast<T *>(x);
}

void _lantern_Tensor_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Tensor>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_bool_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<bool>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_bool_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::bool_t>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_int64_t_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<int64_t>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_vector_int64_t_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::IntArrayRef>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorList_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::TensorList>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_OptionalTensorList_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<c10::List<c10::optional<torch::Tensor>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_double_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<double>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_QScheme_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::QScheme>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Scalar_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Scalar>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScalarType_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::ScalarType>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorOptions_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::TensorOptions>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Dtype_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Dtype>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Device_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::Device>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Layout_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Layout>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Generator_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Generator>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Dimname_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::Dimname>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_DimnameList_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::DimnameList>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_dimname_list_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::DimnameList>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_generator_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::Generator>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_tensor_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::Tensor>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_string_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::string>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_scalar_type_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::ScalarType>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_MemoryFormat_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::MemoryFormat>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_variable_list_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::autograd::variable_list>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::indexing::TensorIndex>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Slice_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::indexing::Slice>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_int64_t_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::int64_t>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_vector_int64_t_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::IntArrayRef>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Function_lambda_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::function<torch::autograd::variable_list(
      torch::autograd::LanternAutogradContext *,
      torch::autograd::variable_list)>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_autograd_edge_list_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::autograd::edge_list>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_autograd_edge_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::autograd::Edge>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_vector_double_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::DoubleArrayRef>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_PackedSequence_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<LanternPtr<torch::nn::utils::rnn::PackedSequence>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Storage_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::Storage>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_const_char_delete(const char *x) {
  LANTERN_FUNCTION_START
  delete[] x;
  LANTERN_FUNCTION_END_VOID
}

void _lantern_IValue_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::jit::IValue>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_vector_string_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<std::string>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_string_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::string>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Stack_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::Stack>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_CompilationUnit_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::CompilationUnit>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_JITModule_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::script::Module>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_TraceableFunction_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<std::function<torch::jit::Stack(torch::jit::Stack)>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_bool_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<Vector<bool>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_void_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<std::vector<void *>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_optional_device_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<self_contained::optional::Device>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_optional_double_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<self_contained::optional::double_t>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_parameter_list_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_parameter_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_module_list_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_module_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_buffer_list_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_parameter_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_ScriptMethod_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::script::Method>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_GenericDict_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<c10::impl::GenericDict>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_GenericList_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<c10::impl::GenericList>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_double_delete(void *x) {
  LANTERN_FUNCTION_START;
  lantern_delete<std::vector<double>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_Tuple_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::IValue>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_TensorDict_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<c10::Dict<std::string, torch::Tensor>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_GenericDict_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<c10::impl::GenericDict>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_GenericList_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<c10::impl::GenericList>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_vector_IValue_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::IValue>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_NamedTupleHelper_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<NamedTupleHelper>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_FunctionPtr_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<torch::jit::Function *>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_vector_Scalar_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::Scalar>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_memory_format_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::MemoryFormat>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_scalar_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::Scalar>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_string_view_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::string_view>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_string_view_delete(void *x) {
  LANTERN_FUNCTION_START
  lantern_delete<self_contained::optional::string_view>(x);
  LANTERN_FUNCTION_END_VOID
}
