#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void* _lantern_QScheme_per_channel_affine() {
  LANTERN_FUNCTION_START
  return make_raw::QScheme(torch::QScheme::PER_CHANNEL_AFFINE);
  LANTERN_FUNCTION_END
}

void* _lantern_QScheme_per_tensor_affine() {
  LANTERN_FUNCTION_START
  return make_raw::QScheme(torch::QScheme::PER_TENSOR_AFFINE);
  LANTERN_FUNCTION_END
}

void* _lantern_QScheme_per_channel_symmetric() {
  LANTERN_FUNCTION_START
  return make_raw::QScheme(torch::QScheme::PER_CHANNEL_SYMMETRIC);
  LANTERN_FUNCTION_END
}

void* _lantern_QScheme_per_tensor_symmetric() {
  LANTERN_FUNCTION_START
  return make_raw::QScheme(torch::QScheme::PER_TENSOR_SYMMETRIC);
  LANTERN_FUNCTION_END
}

const char* _lantern_QScheme_type(void* x) {
  LANTERN_FUNCTION_START
  auto y = from_raw::QScheme(x);

  if (y == torch::QScheme::PER_CHANNEL_AFFINE) {
    return "per_channel_affine";
  }

  if (y == torch::QScheme::PER_TENSOR_AFFINE) {
    return "per_tensor_affine";
  }

  if (y == torch::QScheme::PER_CHANNEL_SYMMETRIC) {
    return "per_channel_symmetric";
  }

  if (y == torch::QScheme::PER_TENSOR_SYMMETRIC) {
    return "per_tensor_symmetric";
  }

  return "not handled";
  LANTERN_FUNCTION_END
}