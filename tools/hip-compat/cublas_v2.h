#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:9 includes
 * <cublas_v2.h> unguarded (hipBLAS itself includes cublas_v2.h only under
 * __HIP_PLATFORM_NVCC__). hipBLAS ships no cublasHandle_t alias. */
#include <hipblas.h>
using cublasHandle_t = hipblasHandle_t;
