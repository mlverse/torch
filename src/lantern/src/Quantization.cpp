#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

bool _lantern_Tensor_is_quantized(void *x) {
  LANTERN_FUNCTION_START
  return from_raw::Tensor(x).is_quantized();
  LANTERN_FUNCTION_END_RET(false)
}
