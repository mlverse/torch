#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v20): ATen/cuda/ATenCUDAGeneral.h:4 and
 * ATen/cuda/CUDADevice.h:6 include <cuda_runtime.h> -- a distinct header from
 * <cuda_runtime_api.h> with the same runtime surface (the light
 * CUDAContextLight.h does not pull it, but adjacent headers do). Forward to
 * the api shim; nothing extra to alias today. */
#include <cuda_runtime_api.h>
