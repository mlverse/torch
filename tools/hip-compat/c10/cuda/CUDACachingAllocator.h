#pragma once
/* c10 -> HIP compat shim (make_rocm_kit.sh v21).
 * The wheel exports c10::hip::HIPCachingAllocator, not the CUDA-named
 * namespace; redirect so the inline get()/getDeviceStats()/emptyCache()
 * wrappers resolve to the wheel's real allocator object. */
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

namespace c10::cuda::CUDACachingAllocator {
using namespace c10::hip::HIPCachingAllocator;
}
