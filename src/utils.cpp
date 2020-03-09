#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_vector_int64_t(int64_t *x, size_t x_size)
{
  auto out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<std::vector<int64_t>>(out);
}

void *lantern_IntArrayRef(int64_t *x, size_t x_size)
{
  torch::IntArrayRef out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<torch::IntArrayRef>(out);
}

void *lantern_int(int x)
{
  return (void *)new LanternObject<int>(x);
}

void *lantern_int64_t(int64_t x)
{
  return (void *)new LanternObject<int64_t>(x);
}

void *lantern_bool(bool x)
{
  return (void *)new LanternObject<bool>(x);
}

void *lantern_vector_get(void *x, int i)
{
  auto v = reinterpret_cast<LanternObject<std::vector<void *>> *>(x)->get();
  return v.at(i);
}