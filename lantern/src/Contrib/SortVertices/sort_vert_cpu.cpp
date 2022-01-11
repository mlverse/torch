#define LANTERN_BUILD
#include <torch/torch.h>

#include "../../utils.hpp"
#include "lantern/lantern.h"

void* _lantern_contrib_sort_vertices(void* vertices, void* mask,
                                     void* num_valid) {
  LANTERN_FUNCTION_START
  throw std::runtime_error(
      "`sort_vertices` is only supported on CUDA runtimes.");
  LANTERN_FUNCTION_END
}