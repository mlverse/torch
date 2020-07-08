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
  lantern_delete<LanternObject<torch::Tensor>>(x);
}

void _lantern_bool_delete(void *x)
{
  lantern_delete<LanternObject<bool>>(x);
}

void _lantern_int64_t_delete(void *x)
{
  lantern_delete<LanternObject<int64_t>>(x);
}

void _lantern_vector_int64_t_delete(void *x)
{
  lantern_delete<LanternObject<std::vector<int64_t>>>(x);
}

void _lantern_TensorList_delete(void *x)
{
  lantern_delete<LanternObject<std::vector<torch::Tensor>>>(x);
}

void _lantern_double_delete(void *x)
{
  lantern_delete<LanternObject<double>>(x);
}

void _lantern_QScheme_delete(void *x)
{
  lantern_delete<LanternObject<torch::QScheme>>(x);
}

void _lantern_Scalar_delete(void *x)
{
  lantern_delete<LanternObject<torch::Scalar>>(x);
}

void _lantern_ScalarType_delete(void *x)
{
  lantern_delete<LanternObject<torch::ScalarType>>(x);
}

void _lantern_TensorOptions_delete(void *x)
{
  lantern_delete<LanternObject<torch::TensorOptions>>(x);
}

void _lantern_Dtype_delete(void *x)
{
  lantern_delete<LanternObject<torch::Dtype>>(x);
}

void _lantern_Device_delete(void *x)
{
  lantern_delete<LanternPtr<torch::Device>>(x);
}

void _lantern_Layout_delete(void *x)
{
  lantern_delete<LanternObject<torch::Layout>>(x);
}

void _lantern_Generator_delete(void *x)
{
  lantern_delete<LanternObject<std::shared_ptr<torch::Generator>>>(x);
}

void _lantern_Dimname_delete(void *x)
{
  lantern_delete<LanternPtr<torch::Dimname>>(x);
}

void _lantern_DimnameList_delete(void *x)
{
  lantern_delete<LanternPtr<std::vector<torch::Dimname>>>(x);
}

void _lantern_MemoryFormat_delete(void *x)
{
  lantern_delete<LanternObject<torch::MemoryFormat>>(x);
}

void _lantern_variable_list_delete(void *x)
{
  lantern_delete<LanternObject<torch::autograd::variable_list>>(x);
}

void _lantern_TensorIndex_delete(void *x)
{
  lantern_delete<LanternObject<std::vector<torch::indexing::TensorIndex>>>(x);
}

void _lantern_Slice_delete(void *x)
{
  lantern_delete<LanternObject<torch::indexing::Slice>>(x);
}

void _lantern_optional_int64_t_delete(void *x)
{
  lantern_delete<LanternObject<c10::optional<int64_t>>>(x);
}

void _lantern_PackedSequence_delete(void *x)
{
  lantern_delete<LanternPtr<torch::nn::utils::rnn::PackedSequence>>(x);
}

void _lantern_Storage_delete(void *x)
{
  lantern_delete<LanternObject<torch::Storage>>(x);
}

void _lantern_const_char_delete (const char * x)
{
  delete []x;
}