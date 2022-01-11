#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

std::int64_t _lantern_Reduction_Mean() {
  LANTERN_FUNCTION_START
  return torch::Reduction::Mean;
  LANTERN_FUNCTION_END_RET(0)
}

std::int64_t _lantern_Reduction_None() {
  LANTERN_FUNCTION_START
  return torch::Reduction::None;
  LANTERN_FUNCTION_END_RET(0)
}

std::int64_t _lantern_Reduction_Sum() {
  LANTERN_FUNCTION_START
  return torch::Reduction::Sum;
  LANTERN_FUNCTION_END_RET(0)
}