#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

bool _lantern_cuda_is_available()
{
    LANTERN_FUNCTION_START
    return torch::cuda::is_available();
    LANTERN_FUNCTION_END_RET(false)
}

int _lantern_cuda_device_count()
{
    LANTERN_FUNCTION_START
    return torch::cuda::device_count();
    LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_cuda_current_device()
{
    LANTERN_FUNCTION_START
    return at::detail::getCUDAHooks().current_device();
    LANTERN_FUNCTION_END_RET(0)
}

void _lantern_cuda_show_config()
{
    LANTERN_FUNCTION_START
    std::cout << at::detail::getCUDAHooks().showConfig() << std::endl;
    LANTERN_FUNCTION_END_VOID
}
