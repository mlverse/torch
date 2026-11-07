#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:8 includes
 * <cusparse.h> unguarded. hipSPARSE ships no cusparse* compat typedefs
 * (hipsparseHandle_t is void*, verified against ROCm 6.4.4). */
#include <hipsparse.h>
using cusparseHandle_t = hipsparseHandle_t;
