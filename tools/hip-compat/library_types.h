#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v20): ATen/cuda/CUDADataType.h:6 includes
 * <library_types.h> for cudaDataType and the CUDA_R_*/CUDA_C_* scalar-type
 * constants. Not in today's lantern closure, but the header sits one include
 * away (CUDABlas.h/CUDASparse*.h). The values mirror CUDA's cublas API
 * (hipBLAS re-exports the same surface as hipDataType/hipblasDatatype_t). */
typedef enum cudaDataType_t {
  CUDA_R_16F = 2,  CUDA_R_32F = 0,  CUDA_R_64F = 1,  CUDA_R_8I = 3,
  CUDA_R_8U = 8,   CUDA_R_32I = 10, CUDA_R_32U = 12, CUDA_R_16BF = 14,
  CUDA_C_16F = 6,  CUDA_C_32F = 4,  CUDA_C_64F = 5,  CUDA_C_8I = 7,
  CUDA_C_32I = 11, CUDA_C_32U = 13, CUDA_C_16BF = 15
} cudaDataType;
