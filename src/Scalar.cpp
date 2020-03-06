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

  return (void *)new LanternObject<torch::optional<torch::Scalar>>(out);
}

void *lantern_Scalar_nullopt()
{
  return (void *)new LanternObject<c10::optional<torch::Scalar>>(c10::nullopt);
}