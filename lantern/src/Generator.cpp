#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Generator()
{
  LANTERN_FUNCTION_START
  torch::Generator out = torch::make_generator<torch::CPUGeneratorImpl>(c10::detail::getNonDeterministicRandom());
  return (void *)new LanternObject<torch::Generator>(out);
  LANTERN_FUNCTION_END
}

void* _lantern_get_default_Generator ()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Generator>(at::detail::getDefaultCPUGenerator());
  LANTERN_FUNCTION_END
}

uint64_t _lantern_Generator_current_seed(void *generator)
{
  LANTERN_FUNCTION_START
  auto gen = reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get();
  return gen.current_seed();
  LANTERN_FUNCTION_END_RET(0)
}

void _lantern_Generator_set_current_seed(void *generator, uint64_t seed)
{
  LANTERN_FUNCTION_START
  reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get().set_current_seed(seed);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_manual_seed (int64_t seed)
{
  torch::manual_seed(seed);
}