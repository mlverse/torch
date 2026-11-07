#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:13 includes
 * <cublasLt.h> unguarded. hipBLASLt defines hipblasLtHandle_t (void*), no
 * cublasLtHandle_t alias. */
#include <hipblaslt.h>
using cublasLtHandle_t = hipblasLtHandle_t;
