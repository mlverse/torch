#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void lantern_Tensor_delete(void *x)
{
  delete reinterpret_cast<LanternObject<torch::Tensor> *>(x);
}