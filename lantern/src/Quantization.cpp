#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

bool _lantern_Tensor_is_quantized(void *x)
{
    LANTERN_FUNCTION_START
    return reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get().is_quantized();
    LANTERN_FUNCTION_END_RET(false)
}
