#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v16): c10/cuda/CUDAException.h:9 includes <cuda.h>
 * unguarded. torch's ROCm build hipifies it to <hip/hip_runtime.h>; the
 * driver-API surface is not referenced by the closure lantern compiles, so
 * the runtime API + the token map below are enough. */
#include <hip/hip_runtime_api.h>
#include <cuda_runtime_api.h>
