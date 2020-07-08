#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

std::int64_t _lantern_Reduction_Mean()
{
    return torch::Reduction::Mean;
}

std::int64_t _lantern_Reduction_None()
{
    return torch::Reduction::None;
}

std::int64_t _lantern_Reduction_Sum()
{
    return torch::Reduction::Sum;
}