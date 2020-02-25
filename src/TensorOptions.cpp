#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_TensorOptions () {
  return (void *) new LanternObject<torch::TensorOptions>(torch::TensorOptions());
}

void* lantern_TensorOptions_dtype (void* self, void* dtype) {
  auto out = reinterpret_cast<torch::TensorOptions * >(self)->dtype(*reinterpret_cast<torch::Dtype *>(dtype));
  return (void *) new LanternObject<torch::TensorOptions>(out);
}

void* lantern_TensorOptions_layout (void* self, void* layout) {
  auto out = reinterpret_cast<torch::TensorOptions * >(self)->layout(*reinterpret_cast<torch::Layout *>(layout));
  return (void *) new LanternObject<torch::TensorOptions>(out);
}

void* lantern_TensorOptions_device (void* self, void* device) {
  auto out = reinterpret_cast<torch::TensorOptions * >(self)->device(*reinterpret_cast<torch::Device *>(device));
  return (void *) new LanternObject<torch::TensorOptions>(out);
}

void* lantern_TensorOptions_requires_grad (void* self, bool requires_grad) {
  auto out = reinterpret_cast<torch::TensorOptions * >(self)->requires_grad(requires_grad);
  return (void *) new LanternObject<torch::TensorOptions>(out);
}

void* lantern_TensorOptions_pinned_memory (void* self, bool pinned_memory) {
  auto out = reinterpret_cast<torch::TensorOptions * >(self)->pinned_memory(pinned_memory);
  return (void *) new LanternObject<torch::TensorOptions>(out);
}