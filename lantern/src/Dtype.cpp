#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Dtype_float32()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat32);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_float64()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat64);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_float16()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kFloat16);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_uint8()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kUInt8);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_int8()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kInt8);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_int16()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kInt16);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_int32()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kInt32);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_int64()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kInt64);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_bool()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kBool);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_quint8()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kQUInt8);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_qint32()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kQInt32);
  LANTERN_FUNCTION_END
}

void *_lantern_Dtype_qint8()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Dtype>(torch::kQInt8);
  LANTERN_FUNCTION_END
}

const char *_lantern_Dtype_type(void *dtype)
{
  LANTERN_FUNCTION_START
  std::string str = toString(reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get());
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void _lantern_set_default_dtype(void *dtype)
{
  LANTERN_FUNCTION_START
  torch::Dtype dt = reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get();
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(dt));
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_get_default_dtype()
{
  LANTERN_FUNCTION_START
  auto dt = torch::get_default_dtype();
  return (void *)new LanternObject<torch::Dtype>(c10::typeMetaToScalarType(dt));
  LANTERN_FUNCTION_END
}
