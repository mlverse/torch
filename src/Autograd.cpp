#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void lantern_autograd_set_grad_mode(bool enabled)
{
    torch::autograd::GradMode::set_enabled(enabled);
}
