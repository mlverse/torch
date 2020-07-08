#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Dtype_float32()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat32);
}

void *_lantern_Dtype_float64()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat64);
}

void *_lantern_Dtype_float16()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat16);
}

void *_lantern_Dtype_uint8()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kUInt8);
}

void *_lantern_Dtype_int8()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kInt8);
}

void *_lantern_Dtype_int16()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kInt16);
}

void *_lantern_Dtype_int32()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kInt32);
}

void *_lantern_Dtype_int64()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kInt64);
}

void *_lantern_Dtype_bool()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kBool);
}

void *_lantern_Dtype_quint8()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kQUInt8);
}

void *_lantern_Dtype_qint32()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kQInt32);
}

void *_lantern_Dtype_qint8()
{
  return (void *)new LanternObject<torch::Dtype>(torch::kQInt8);
}

const char *_lantern_Dtype_type(void *dtype)
{
  return toString(reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get());
}

void _lantern_set_default_dtype(void *dtype)
{
  torch::Dtype dt = reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get();
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(dt));
}

void *_lantern_get_default_dtype()
{
  auto dt = torch::get_default_dtype();
  return (void *)new LanternObject<torch::Dtype>(c10::typeMetaToScalarType(dt));
}
