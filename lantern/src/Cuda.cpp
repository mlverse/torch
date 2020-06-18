#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

bool lantern_cuda_is_available()
{
    return torch::cuda::is_available();
}

int lantern_cuda_device_count()
{
    return torch::cuda::device_count();
}

int64_t lantern_cuda_current_device()
{
    return at::detail::getCUDAHooks().current_device();
}

void lantern_cuda_show_config()
{
    std::cout << at::detail::getCUDAHooks().showConfig() << std::endl;
}
