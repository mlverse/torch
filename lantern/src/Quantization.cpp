#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

bool lantern_Tensor_is_quantized(void *x)
{
    return reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get().is_quantized();
}
