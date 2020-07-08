#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Scalar(void *value, const char *type)
{
  LANTERN_FUNCTION_START
  torch::Scalar out;
  auto TYPE = std::string(type);

  if (TYPE == "int")
  {
    out = *reinterpret_cast<int *>(value);
  }
  else if (TYPE == "bool")
  {
    out = *reinterpret_cast<bool *>(value);
  }
  else if (TYPE == "double")
  {
    out = *reinterpret_cast<double *>(value);
  }

  return (void *)new LanternObject<torch::Scalar>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_Scalar_dtype(void *self)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return (void *)new LanternObject<torch::Dtype>(v.type());
  LANTERN_FUNCTION_END
}

float _lantern_Scalar_to_float(void *self)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toFloat();
  LANTERN_FUNCTION_END_RET(0.0f)
}

int _lantern_Scalar_to_int(void *self)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toInt();
  LANTERN_FUNCTION_END_RET(0)
}

double _lantern_Scalar_to_double(void *self)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toDouble();
  LANTERN_FUNCTION_END_RET(0)
}

bool _lantern_Scalar_to_bool(void *self)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toBool();
  LANTERN_FUNCTION_END_RET(false)
}

void *_lantern_Scalar_nullopt()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<c10::optional<torch::Scalar>>(c10::nullopt);
  LANTERN_FUNCTION_END
}