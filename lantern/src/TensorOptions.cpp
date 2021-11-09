#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_TensorOptions()
{
  LANTERN_FUNCTION_START
  return make_unique::TensorOptions(torch::TensorOptions());
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_dtype(void *self, void *dtype)
{
  LANTERN_FUNCTION_START
  auto out = from_raw::TensorOptions(self).dtype(reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get());
  return make_unique::TensorOptions(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_layout(void *self, void *layout)
{
  LANTERN_FUNCTION_START
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().layout(reinterpret_cast<LanternObject<torch::Layout> *>(layout)->get());
  return (void *)new LanternObject<torch::TensorOptions>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_device(void *self, void *device)
{
  LANTERN_FUNCTION_START
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().device(((LanternPtr<torch::Device> *)device)->get());
  return (void *)new LanternObject<torch::TensorOptions>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_requires_grad(void *self, bool requires_grad)
{
  LANTERN_FUNCTION_START
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().requires_grad(requires_grad);
  return (void *)new LanternObject<torch::TensorOptions>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_TensorOptions_pinned_memory(void *self, bool pinned_memory)
{
  LANTERN_FUNCTION_START
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().pinned_memory(pinned_memory);
  return (void *)new LanternObject<torch::TensorOptions>(out);
  LANTERN_FUNCTION_END
}

void _lantern_TensorOptions_print(void *self)
{
  LANTERN_FUNCTION_START
  std::cout << reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get() << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorOptions_address(void *self)
{
  LANTERN_FUNCTION_START
  std::cout << &(reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get()) << std::endl;
  LANTERN_FUNCTION_END_VOID
}