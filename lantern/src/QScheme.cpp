#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_QScheme_per_channel_affine () {
  return (void *) new LanternObject<torch::QScheme>(torch::QScheme::PER_CHANNEL_AFFINE);
}

void* lantern_QScheme_per_tensor_affine () {
  return (void *) new LanternObject<torch::QScheme>(torch::QScheme::PER_TENSOR_AFFINE);
}

void* lantern_QScheme_per_channel_symmetric () {
  return (void *) new LanternObject<torch::QScheme>(torch::QScheme::PER_CHANNEL_SYMMETRIC);
}

void* lantern_QScheme_per_tensor_symmetric () {
  return (void *) new LanternObject<torch::QScheme>(torch::QScheme::PER_TENSOR_SYMMETRIC);
}

const char * lantern_QScheme_type(void* x) {
  
  torch::QScheme y = reinterpret_cast<LanternObject<torch::QScheme>*>(x)->get();
  
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
}