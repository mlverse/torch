#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v19): ATen/cuda/CUDAContextLight.h:9 includes
 * <cublas_v2.h> unguarded (hipBLAS itself includes cublas_v2.h only under
 * __HIP_PLATFORM_NVCC__). hipBLAS ships no cublasHandle_t alias. */
#include <hipblas.h>
using cublasHandle_t = hipblasHandle_t;
/* ATen/cuda/Exceptions.h:59 declares _cublasGetErrorEnum(cublasStatus_t);
 * alias the STATUS type too, and CUBLAS_STATUS_SUCCESS for the
 * AT_CUBLAS_CHECK macro (hipblasStatus_t is an enum;
 * HIPBLAS_STATUS_SUCCESS ships in every hipblas version). */
using cublasStatus_t = hipblasStatus_t;
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
