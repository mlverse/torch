#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_vector_int64_t (int64_t* x, size_t x_size) {
  auto out = std::vector<int64_t>(x, x + x_size);
  return (void *) new LanternObject<std::vector<int64_t>>(out);
}

void* lantern_IntArrayRef (int64_t* x, size_t x_size) {
  torch::IntArrayRef out = std::vector<int64_t>(x, x + x_size);
  return (void *) new LanternObject<torch::IntArrayRef>(out);
}