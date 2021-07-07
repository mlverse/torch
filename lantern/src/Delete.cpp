#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

template <class T>
void lantern_delete(void *x)
{
  delete reinterpret_cast<T *>(x);
}

void _lantern_Tensor_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Tensor>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_bool_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<bool>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_int64_t_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<int64_t>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_vector_int64_t_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<std::vector<int64_t>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorList_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<std::vector<torch::Tensor>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_double_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<double>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_QScheme_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::QScheme>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Scalar_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Scalar>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_ScalarType_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::ScalarType>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorOptions_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::TensorOptions>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Dtype_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Dtype>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Device_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternPtr<torch::Device>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Layout_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Layout>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Generator_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Generator>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Dimname_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternPtr<torch::Dimname>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_DimnameList_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternPtr<std::vector<torch::Dimname>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_MemoryFormat_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::MemoryFormat>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_variable_list_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::autograd::variable_list>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<std::vector<torch::indexing::TensorIndex>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Slice_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::indexing::Slice>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_int64_t_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<c10::optional<int64_t>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_optional_vector_int64_t_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<c10::optional<torch::ArrayRef<int64_t>>>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_PackedSequence_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternPtr<torch::nn::utils::rnn::PackedSequence>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Storage_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<torch::Storage>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_const_char_delete (const char * x)
{
  LANTERN_FUNCTION_START
  delete []x;
  LANTERN_FUNCTION_END_VOID
}

void _lantern_IValue_delete (void * x)
{
  LANTERN_FUNCTION_START
  lantern_delete<torch::jit::IValue>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_vector_string_delete (void * x)
{
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<std::string>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_string_delete (void * x)
{
  LANTERN_FUNCTION_START
  lantern_delete<LanternObject<std::string>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Stack_delete (void * x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<LanternObject<torch::jit::Stack>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_CompilationUnit_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::CompilationUnit>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_JITModule_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::script::Module>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_TraceableFunction_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<LanternObject<std::function<torch::jit::Stack(torch::jit::Stack)>>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_bool_delete(void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<std::vector<bool>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_void_delete(void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<std::vector<void*>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_optional_tensor_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<LanternObject<c10::optional<torch::Tensor>>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_optional_device_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<LanternObject<c10::optional<torch::Device>>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_parameter_list_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_parameter_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_module_list_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_module_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_named_buffer_list_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::named_parameter_list>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_jit_ScriptMethod_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<torch::jit::script::Method>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_GenericDict_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<c10::impl::GenericDict>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_GenericList_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<c10::impl::GenericList>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_double_delete (void* x)
{
  LANTERN_FUNCTION_START;
  lantern_delete<std::vector<double>>(x);
  LANTERN_FUNCTION_END_VOID;
}

void _lantern_vector_int64_t2_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<int64_t>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_Tuple_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::IValue>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_TensorDict_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<c10::Dict<std::string, torch::Tensor>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_GenericDict_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<c10::impl::GenericDict>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_GenericList_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<c10::impl::GenericList>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_jit_vector_IValue_delete(void *x)
{
  LANTERN_FUNCTION_START
  lantern_delete<std::vector<torch::IValue>>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_NamedTupleHelper_delete (void* x)
{
  LANTERN_FUNCTION_START
  lantern_delete<NamedTupleHelper>(x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_FunctionPtr_delete (void* x)
{
  LANTERN_FUNCTION_START
  lantern_delete<torch::jit::Function*>(x);
  LANTERN_FUNCTION_END_VOID
}

