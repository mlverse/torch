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

void lantern_Tensor_delete(void *x)
{
  lantern_delete<LanternObject<torch::Tensor>>(x);
}

void lantern_bool_delete(void *x)
{
  lantern_delete<LanternObject<bool>>(x);
}

void lantern_int64_t_delete(void *x)
{
  lantern_delete<LanternObject<int64_t>>(x);
}

void lantern_vector_int64_t_delete(void *x)
{
  lantern_delete<LanternObject<std::vector<int64_t>>>(x);
}

void lantern_TensorList_delete(void *x)
{
  lantern_delete<LanternObject<std::vector<torch::Tensor>>>(x);
}

void lantern_double_delete(void *x)
{
  lantern_delete<LanternObject<double>>(x);
}

void lantern_QScheme_delete(void *x)
{
  lantern_delete<LanternObject<torch::QScheme>>(x);
}

void lantern_Scalar_delete(void *x)
{
  lantern_delete<LanternObject<torch::Scalar>>(x);
}

void lantern_ScalarType_delete(void *x)
{
  lantern_delete<LanternObject<torch::ScalarType>>(x);
}

void lantern_TensorOptions_delete(void *x)
{
  lantern_delete<LanternObject<torch::TensorOptions>>(x);
}

void lantern_Dtype_delete(void *x)
{
  lantern_delete<LanternObject<torch::Dtype>>(x);
}

void lantern_Device_delete(void *x)
{
  lantern_delete<LanternPtr<torch::Device>>(x);
}

void lantern_Layout_delete(void *x)
{
  lantern_delete<LanternObject<torch::Layout>>(x);
}

void lantern_Generator_delete(void *x)
{
  lantern_delete<LanternObject<std::shared_ptr<torch::Generator>>>(x);
}

void lantern_Dimname_delete(void *x)
{
  lantern_delete<LanternPtr<torch::Dimname>>(x);
}

void lantern_DimnameList_delete(void *x)
{
  lantern_delete<LanternPtr<std::vector<torch::Dimname>>>(x);
}

void lantern_MemoryFormat_delete(void *x)
{
  lantern_delete<LanternObject<torch::MemoryFormat>>(x);
}

void lantern_variable_list_delete(void *x)
{
  lantern_delete<LanternObject<torch::autograd::variable_list>>(x);
}