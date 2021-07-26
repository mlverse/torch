#define LANTERN_BUILD
#include "lantern/lantern.h"
#include "utils.h"
#include <torch/torch.h>
#include "../../utils.hpp"


void* _lantern_contrib_sort_vertices (void* vertices, void* mask, void* num_valid)
{
    throw std::runtime_error("`sort_vertices` is only supported on CUDA runtimes.");
}