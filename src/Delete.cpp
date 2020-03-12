#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

template <class T>
void lantern_delete(void *x)
{
  delete reinterpret_cast<LanternObject<T> *>(x);
}

void lantern_Tensor_delete(void *x)
{
  lantern_delete<torch::Tensor>(x);
}

void lantern_bool_delete(void *x)
{
  lantern_delete<bool>(x);
}

void lantern_int64_t_delete(void *x)
{
  lantern_delete<int64_t>(x);
}

void lantern_TensorList_delete(void *x)
{
  lantern_delete<std::vector<torch::Tensor>>(x);
}

void lantern_double_delete(void *x)
{
  lantern_delete<double>(x);
}

void lantern_QScheme_delete(void *x)
{
  lantern_delete<torch::QScheme>(x);
}

void lantern_Scalar_delete(void *x)
{
  lantern_delete<torch::Scalar>(x);
}

void lantern_ScalarType_delete(void *x)
{
  lantern_delete<torch::ScalarType>(x);
}

void lantern_TensorOptions_delete(void *x)
{
  lantern_delete<torch::TensorOptions>(x);
}

void lantern_Dtype_delete(void *x)
{
  lantern_delete<torch::Dtype>(x);
}

void lantern_Device_delete(void *x)
{
  lantern_delete<torch::Device>(x);
}

void lantern_Layout_delete(void* x)
{
  lantern_delete<torch::Layout>(x);
}