#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_TensorOptions() {
  LANTERN_FUNCTION_START
  return make_raw::TensorOptions(torch::TensorOptions());
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_dtype(void *self, void *dtype) {
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).dtype(from_raw::Dtype(dtype));
  return make_raw::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_layout(void *self, void *layout) {
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).layout(from_raw::Layout(layout));
  return make_raw::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_device(void *self, void *device) {
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).device(from_raw::Device(device));
  return make_raw::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_requires_grad(void *self, bool requires_grad) {
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).requires_grad(requires_grad);
  return make_raw::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_pinned_memory(void *self, bool pinned_memory) {
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).pinned_memory(pinned_memory);
  return make_raw::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void _lantern_TensorOptions_print(void *self) {
  LANTERN_FUNCTION_START
  std::cout << from_raw::TensorOptions(self) << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorOptions_address(void *self) {
  LANTERN_FUNCTION_START
  std::cout << &(from_raw::TensorOptions(self)) << std::endl;
  LANTERN_FUNCTION_END_VOID
}