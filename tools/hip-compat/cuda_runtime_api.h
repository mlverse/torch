#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v16).
 * torch 2.8.0 includes <cuda_runtime_api.h> UNGUARDED even under USE_ROCM
 * (ATen/cuda/CUDAContextLight.h:7, c10/cuda/CUDAStream.h:3,
 * c10/cuda/CUDAFunctions.h:14). torch's own ROCm build hipifies its sources
 * before compiling; downstream g++ builds must map the CUDA names onto HIP
 * themselves. Token set = the exact cuda* closure of the torch 2.8.0 headers
 * lantern compiles (ATen/cuda/CUDAContext.h, ATen/cuda/detail/CUDAHooks.h,
 * c10/cuda/CUDACachingAllocator.h + their c10/cuda closure). */
#include <hip/hip_runtime_api.h>

using cudaError_t    = hipError_t;
using cudaStream_t   = hipStream_t;
using cudaEvent_t    = hipEvent_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStreamCaptureMode   = hipStreamCaptureMode;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;

#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyAsyncPeer hipMemcpyPeerAsync
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaEventCreateWithFlags hipEventCreateWithFlags
