#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_Scalar(void *value, const char *type)
{
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
}

void *lantern_Scalar_dtype(void *self)
{
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return (void *)new LanternObject<torch::Dtype>(v.type());
}

float lantern_Scalar_to_float(void *self)
{
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toFloat();
}

int lantern_Scalar_to_int(void *self)
{
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toInt();
}

double lantern_Scalar_to_double(void *self)
{
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toDouble();
}

bool lantern_Scalar_to_bool(void *self)
{
  auto v = reinterpret_cast<LanternObject<torch::Scalar> *>(self)->get();
  return v.toBool();
}

void *lantern_Scalar_nullopt()
{
  return (void *)new LanternObject<c10::optional<torch::Scalar>>(c10::nullopt);
}