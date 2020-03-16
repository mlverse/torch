#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_Generator()
{
  std::shared_ptr<torch::Generator> out = std::shared_ptr<torch::Generator>(new at::CPUGenerator());
  return (void *)new LanternObject<std::shared_ptr<torch::Generator>>(out);
}

uint64_t lantern_Generator_current_seed(void *generator)
{
  return reinterpret_cast<LanternObject<std::shared_ptr<torch::Generator>> *>(generator)->get()->current_seed();
}

void lantern_Generator_set_current_seed(void *generator, uint64_t seed)
{
  reinterpret_cast<LanternObject<std::shared_ptr<torch::Generator>> *>(generator)->get()->set_current_seed(seed);
}