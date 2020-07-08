#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_TensorOptions()
{
  return (void *)new LanternObject<torch::TensorOptions>(torch::TensorOptions());
}

void *_lantern_TensorOptions_dtype(void *self, void *dtype)
{
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().dtype(reinterpret_cast<LanternObject<torch::Dtype> *>(dtype)->get());
  return (void *)new LanternObject<torch::TensorOptions>(out);
}

void *_lantern_TensorOptions_layout(void *self, void *layout)
{
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().layout(reinterpret_cast<LanternObject<torch::Layout> *>(layout)->get());
  return (void *)new LanternObject<torch::TensorOptions>(out);
}

void *_lantern_TensorOptions_device(void *self, void *device)
{
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().device(((LanternPtr<torch::Device> *)device)->get());
  return (void *)new LanternObject<torch::TensorOptions>(out);
}

void *_lantern_TensorOptions_requires_grad(void *self, bool requires_grad)
{
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().requires_grad(requires_grad);
  return (void *)new LanternObject<torch::TensorOptions>(out);
}

void *_lantern_TensorOptions_pinned_memory(void *self, bool pinned_memory)
{
  auto out = reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get().pinned_memory(pinned_memory);
  return (void *)new LanternObject<torch::TensorOptions>(out);
}

void _lantern_TensorOptions_print(void *self)
{
  std::cout << reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get() << std::endl;
}

void _lantern_TensorOptions_address(void *self)
{
  std::cout << &(reinterpret_cast<LanternObject<torch::TensorOptions> *>(self)->get()) << std::endl;
}