#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Generator()
{
  LANTERN_FUNCTION_START
  std::shared_ptr<torch::Generator> out = std::make_shared<at::CPUGenerator>();
  return (void *)new LanternObject<std::shared_ptr<torch::Generator>>(out);
  LANTERN_FUNCTION_END
}

uint64_t _lantern_Generator_current_seed(void *generator)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<LanternObject<std::shared_ptr<torch::Generator>> *>(generator)->get()->current_seed();
  LANTERN_FUNCTION_END_RET(0)
}

void _lantern_Generator_set_current_seed(void *generator, uint64_t seed)
{
  LANTERN_FUNCTION_START
  reinterpret_cast<LanternObject<std::shared_ptr<torch::Generator>> *>(generator)->get()->set_current_seed(seed);
  LANTERN_FUNCTION_END_VOID
}