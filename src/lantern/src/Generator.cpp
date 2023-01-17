#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_Generator() {
  LANTERN_FUNCTION_START
  auto out = torch::make_generator<torch::CPUGeneratorImpl>(
      c10::detail::getNonDeterministicRandom());
  return make_raw::Generator(out);
  LANTERN_FUNCTION_END
}

void *_lantern_get_default_Generator() {
  LANTERN_FUNCTION_START
  return make_raw::Generator(at::detail::getDefaultCPUGenerator());
  LANTERN_FUNCTION_END
}

uint64_t _lantern_Generator_current_seed(void *generator) {
  LANTERN_FUNCTION_START
  auto gen = from_raw::Generator(generator);
  return gen.current_seed();
  LANTERN_FUNCTION_END_RET(0)
}

void _lantern_Generator_set_current_seed(void *generator, uint64_t seed) {
  LANTERN_FUNCTION_START
  from_raw::Generator(generator).set_current_seed(seed);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_manual_seed(int64_t seed) { torch::manual_seed(seed); }